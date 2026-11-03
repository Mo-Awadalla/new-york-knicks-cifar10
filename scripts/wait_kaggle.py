"""Wait for a Kaggle kernel and download its actual output artifacts."""

import argparse
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(["kaggle", "kernels", "status", args.kernel], text=True, capture_output=True, check=True, timeout=60)
        status = result.stdout.strip()
        print(status, flush=True)
        if '"KernelWorkerStatus.COMPLETE"' in status or '"KernelWorkerStatus.ERROR"' in status or '"KernelWorkerStatus.CANCELLED"' in status:
            subprocess.run(["kaggle", "kernels", "output", args.kernel, "-p", str(args.output)], check=True, timeout=300)
            if '"KernelWorkerStatus.COMPLETE"' not in status:
                raise SystemExit("Kaggle kernel did not complete successfully. Inspect downloaded logs.")
            return
        if not any(value in status for value in ('"KernelWorkerStatus.RUNNING"', '"KernelWorkerStatus.QUEUED"')):
            raise SystemExit(f"Unexpected Kaggle kernel state: {status}")
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise SystemExit("Wait deadline reached. The remote kernel has not been cancelled.")


if __name__ == "__main__":
    main()
