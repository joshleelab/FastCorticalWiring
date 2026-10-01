#!/usr/bin/env python3
"""Parallel native-space FastCW runner (no resampling) with NUMA Pinning."""

import argparse
from datetime import datetime, timezone
import logging
import os
import subprocess
import sys
import queue
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

ENV = {}

def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def get_numa_balanced_cores(target_workers, force_logical=False):
    """
    Prefer one logical CPU per physical core.
    If target_workers exceeds the number of physical cores, only use sibling
    hyperthreads when force_logical=True.
    """
    sockets_primary = {}
    sockets_all = {}

    for cpu in range(psutil.cpu_count(logical=True)):
        try:
            with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list', 'r') as f:
                sib_text = f.read().strip()

            # Expand formats like "0,64" or "0-1,64-65"
            sibs = []
            for part in sib_text.split(','):
                if '-' in part:
                    a, b = map(int, part.split('-'))
                    sibs.extend(range(a, b + 1))
                else:
                    sibs.append(int(part))

            primary_thread = min(sibs)

            with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id', 'r') as f:
                socket_id = int(f.read().strip())

            sockets_all.setdefault(socket_id, []).append(cpu)
            if cpu == primary_thread:
                sockets_primary.setdefault(socket_id, []).append(cpu)

        except IOError:
            continue

    primary_cores = sum(len(v) for v in sockets_primary.values())
    logical_cores = sum(len(v) for v in sockets_all.values())

    if target_workers <= primary_cores:
        candidate_sockets = sockets_primary
    else:
        if not force_logical:
            raise ValueError(
                f"Requested {target_workers} workers, but only {primary_cores} physical cores "
                f"available. Re-run with --force-logical to use up to {logical_cores} logical CPUs."
            )
        if target_workers > logical_cores:
            raise ValueError(
                f"Requested {target_workers} workers, but only {logical_cores} logical CPUs available."
            )
        candidate_sockets = sockets_all

    assigned_cores = []
    num_sockets = len(candidate_sockets)
    if num_sockets == 0:
        return list(range(target_workers))

    workers_per_socket = {sock: 0 for sock in candidate_sockets}
    for i in range(target_workers):
        sock = sorted(candidate_sockets.keys())[i % num_sockets]
        workers_per_socket[sock] += 1

    for sock in sorted(candidate_sockets.keys()):
        cores = sorted(candidate_sockets[sock])
        needed = workers_per_socket[sock]
        if needed == 0:
            continue

        if needed > len(cores):
            raise ValueError(
                f"Socket {sock}: requested {needed} workers but only {len(cores)} CPUs available."
            )

        if needed == 1:
            assigned_cores.append(cores[0])
        else:
            step = (len(cores) - 1) / (needed - 1)
            for i in range(needed):
                assigned_cores.append(cores[round(i * step)])

    return assigned_cores

def run_cmd(cmd, subject_id, core_id, log_file=None, dry_run=False):
    """Execute a command, strictly pinned to a specific core."""
    if dry_run:
        logging.info(f"[DRY RUN - {subject_id} on Core {core_id}] {' '.join(cmd)}")
        return True

    # preexec_fn sets the CPU affinity of the child process BEFORE it loads Python
    def set_affinity():
        os.sched_setaffinity(0, {core_id})

    try:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{_utc_now_iso()}] START subject={subject_id} core={core_id}\n")
                f.write(f"[{_utc_now_iso()}] CMD {' '.join(cmd)}\n")
                f.flush()
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=ENV, preexec_fn=set_affinity)
                f.write(f"[{_utc_now_iso()}] FINISH subject={subject_id} status=ok\n")
                f.flush()
        else:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=ENV, preexec_fn=set_affinity)
        return True
    except subprocess.CalledProcessError as exc:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{_utc_now_iso()}] FINISH subject={subject_id} status=failed returncode={exc.returncode}\n")
                f.flush()
        error_msg = exc.stderr if exc.stderr else "Check log file for details."
        logging.error(f"[{subject_id}] Command failed on Core {core_id}: {' '.join(cmd[:3])}... \nError: {error_msg}")
        return False

def build_fastcw_cmd(args, subject):
    """Build FastCW command for a single subject in native surface space."""
    cmd = [
        args.venv_python, args.fastcw_path,
        "--surf-type", args.surf_type,
        "--hemispheres", *args.hemispheres,
    ]

    if args.custom_label: cmd.extend(["--custom-label", args.custom_label])
    if args.mask: cmd.extend(["--mask", args.mask])
    if args.no_mask: cmd.append("--no-mask")
    if args.overwrite: cmd.append("--overwrite")
    cmd.extend(["--output-label", args.output_label])
    if args.output_dir: cmd.extend(["--output-dir", args.output_dir])
    if args.output_format: cmd.extend(["--output-format", args.output_format])
    if args.engine: cmd.extend(["--engine", args.engine])
    diffusion_length = 0.7 if args.diffusion_length_mm is None else args.diffusion_length_mm
    if args.diffusion_length_mm is None:
        for item in args.engine_kw:
            key, sep, value = item.partition("=")
            if sep and key.strip() == "diffusion_length_mm":
                diffusion_length = float(value)
    cmd.extend(["--diffusion-length-mm", str(diffusion_length)])
    if args.sample is not None:
        cmd.extend(["--sample", str(args.sample)])
        if args.sample_kind is not None:
            cmd.extend(["--sample-kind", str(args.sample_kind)])
    if args.sample_method is not None: cmd.extend(["--sample-method", args.sample_method])
    if args.vertex_list: cmd.extend(["--vertex-list", args.vertex_list])
    if args.verbose_timing: cmd.append("--verbose-timing")

    cmd.extend([
        "--scale", *[str(s) for s in args.scale],
        "--eps", str(args.eps),
        "--n-samples-between-scales", str(args.n_samples_between_scales),
        "--boundary-cap-fraction", "none" if args.boundary_cap_fraction is None else str(args.boundary_cap_fraction),
    ])
    for item in args.engine_kw:
        cmd.extend(["--engine-kw", item])

    cmd.extend([args.subjects_dir, subject])
    return cmd

def process_subject(subject, args, core_queue):
    """Run FastCW for one subject pinned to an assigned core."""
    core_id = core_queue.get()
    logging.info(f"STARTING: {subject} (Assigned to Core {core_id})")

    log_path = Path(args.log_dir) / f"{subject}.log"
    if args.overwrite and log_path.exists() and not args.dry_run:
        log_path.unlink()

    cmd = build_fastcw_cmd(args, subject)
    success = run_cmd(cmd, subject, core_id, log_file=log_path, dry_run=args.dry_run)

    if success:
        logging.info(f"FINISHED: {subject} (Core {core_id} released)")
    else:
        logging.error(f"FAILED: {subject} (Core {core_id} released)")

    core_queue.put(core_id)

def main():
    parser = argparse.ArgumentParser(description="Parallel native-space FastCW pipeline (NUMA aware)")
    parser.add_argument("subject_list", help="Path to text file containing subject IDs")
    parser.add_argument("--subjects-dir", default=os.environ.get("SUBJECTS_DIR"), help="Path to FreeSurfer SUBJECTS_DIR")
    parser.add_argument("--fastcw-path", default=str(Path(__file__).parent.resolve() / "fastcw.py"), help="Path to the FastCW CLI script")
    parser.add_argument("--venv-python", default=sys.executable, help="Path to Python executable")
    parser.add_argument("--surf-type", default="pial", help="Native surface type for FastCW")
    parser.add_argument("--custom-label", default=None, help="Optional custom cortex label name")
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"], help="Hemispheres to process")
    parser.add_argument("--mask", default=None, help="Optional explicit mask path")
    parser.add_argument("--no-mask", action="store_true", help="Run FastCW without cortical masking")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    parser.add_argument(
        "--output-label",
        required=True,
        help="Required FastCW run label; default outputs go under each subject's surf/<label>/ directory",
    )
    parser.add_argument("--output-format", default=None, help="Optional FastCW output format override")
    parser.add_argument("--engine", default=None, help="Optional FastCW geodesic engine")
    parser.add_argument(
        "--diffusion-length-mm",
        type=float,
        default=None,
        help="Heat-method diffusion length sqrt(t) in mm (potpourri engine; default 0.7)",
    )
    # Deprecated no-ops, accepted so old command lines still run (robust mode is always on;
    # sources are solved one at a time; radii are solved exactly).
    parser.add_argument("--use-robust", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--force-logical",
        action="store_true",
        help="Allow use of hyperthreads/logical sibling CPUs when jobs exceed physical cores",)
    parser.add_argument("--engine-kw", action="append", default=[], help="FastCW engine-specific key=value (repeatable)")
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Primary sampling size control (fraction by default; use --sample-kind count for exact cardinality)",
    )
    parser.add_argument(
        "--sample-kind",
        choices=["frac", "count"],
        default="frac",
        help="Interpretation of --sample (default: frac)",
    )
    parser.add_argument(
        "--sample-method",
        choices=["stratified", "random", "fps"],
        default=None,
        help="Sampling method (defaults to stratified when sampling is enabled)",
    )
    parser.add_argument("--vertex-list", default=None, help="Subset by vertex-index list file")
    parser.add_argument(
        "--scale",
        nargs="+",
        type=float,
        default=[
            0.00200000,
            0.00267988,
            0.00359088,
            0.00481157,
            0.00644721,
            0.00863888,
            0.01157558,
            0.01551059,
            0.02078326,
            0.02784833,
            0.03731509,
            0.05000000,
        ],
        help="One or more scales for local measures",
    )
    parser.add_argument("--area-tol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical tolerance for isoline tests")
    parser.add_argument(
        "--n-samples-between-scales",
        type=int,
        default=3,
        help="Supplementary log-spaced area samples between adjacent solved scale radii",
    )
    parser.add_argument(
        "--boundary-cap-fraction",
        type=lambda raw: None if str(raw).strip().lower() in {"none", "null", "off", "false", "disable", "disabled"} else float(raw),
        default=None,
        help="Skip supplementary samples whose estimated boundary-clipped disc area fraction exceeds this value (0.05 = 5 percent area-loss tolerance); unset or 'none' disables the cap",
    )
    parser.add_argument(
        "--verbose-timing",
        action="store_true",
        default=False,
        help="Emit FastCW per-vertex timing lines",
    )
    parser.add_argument("-j", "--jobs", type=int, default=16, help="Number of parallel processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--log-dir", default="logs_fastcw", help="Directory for per-subject logs")
    args = parser.parse_args()

    if args.use_robust:
        print("NOTE: --use-robust is deprecated and ignored; potpourri3d always runs in robust mode.")
    if args.batch_size is not None:
        print("NOTE: --batch-size is deprecated and ignored; sources are solved one at a time.")
    if args.area_tol is not None:
        print("NOTE: --area-tol is deprecated and ignored; radii are solved exactly.")

    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    if not args.subjects_dir:
        logging.error("SUBJECTS_DIR is not set and was not provided via --subjects-dir.")
        sys.exit(1)
    if not Path(args.subject_list).exists():
        logging.error(f"Subject list '{args.subject_list}' not found.")
        sys.exit(1)
    if not Path(args.fastcw_path).exists():
        logging.error(f"FastCW script '{args.fastcw_path}' not found.")
        sys.exit(1)

    global ENV
    ENV = os.environ.copy()
    ENV.update({
        "SUBJECTS_DIR": args.subjects_dir,
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"
    })

    with open(args.subject_list, "r", encoding="utf-8") as f:
        subjects = [line.strip() for line in f if line.strip()]

    # Hardware Setup
    balanced_cores = get_numa_balanced_cores(args.jobs, force_logical=args.force_logical)
    core_queue = queue.Queue()
    for core in balanced_cores:
        core_queue.put(core)

    mode = "DRY RUN" if args.dry_run else "EXECUTION"
    logging.info(f"Initializing {mode} for {len(subjects)} subjects using {args.jobs} workers.")
    logging.info(f"NUMA Pinning Map: {balanced_cores}")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(process_subject, sub, args, core_queue) for sub in subjects]
        for future in as_completed(futures):
            # Capture any unexpected thread exceptions
            try:
                future.result()
            except Exception as e:
                logging.error(f"Pipeline thread crashed: {e}")

    logging.info("Pipeline complete.")

if __name__ == "__main__":
    main()
