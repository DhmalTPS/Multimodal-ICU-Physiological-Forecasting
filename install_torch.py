"""
install_torch.py -- auto-detecting PyTorch installer.

Why this exists: `pip install torch==2.2.2` alone resolves inconsistently across machines
(CPU wheel on some setups, a fixed CUDA build on others), and a hardcoded --index-url in
requirements.txt breaks the moment this project runs on a machine with a different CUDA
version, no GPU, or a different OS. This script asks the machine what it actually has and
installs the matching wheel -- run it once, before `pip install -r requirements.txt`.

Usage:
    python install_torch.py            # auto-detect and install
    python install_torch.py --check    # only report what would be installed, don't install
"""
import re
import shutil
import subprocess
import sys

TORCH_VERSION = "2.2.2"

# nvidia-smi's reported "CUDA Version" is the MAX version the driver supports, not a
# requirement to match exactly -- a driver reporting 12.4 works fine with a cu121 wheel.
# We therefore only need two buckets: "12.x driver" -> cu121 wheel, "11.x driver" -> cu118 wheel.
CUDA_BUCKETS = [
    (12, 0, "cu121", "https://download.pytorch.org/whl/cu121"),
    (11, 0, "cu118", "https://download.pytorch.org/whl/cu118"),
]


def detect_driver_cuda_version() -> tuple[int, int] | None:
    """Returns (major, minor) of the CUDA version reported by nvidia-smi, or None if
    nvidia-smi is not on PATH (no NVIDIA driver -> no usable GPU for PyTorch)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*([\d]+)\.([\d]+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pick_install_command(cuda_ver: tuple[int, int] | None) -> list[str]:
    if cuda_ver is None:
        print("No NVIDIA driver detected (nvidia-smi not found) -> installing CPU-only PyTorch.")
        return [sys.executable, "-m", "pip", "install", f"torch=={TORCH_VERSION}+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"]

    major, minor = cuda_ver
    for req_major, req_minor, tag, index_url in CUDA_BUCKETS:
        if (major, minor) >= (req_major, req_minor):
            print(f"Driver reports CUDA {major}.{minor} -> installing PyTorch {TORCH_VERSION} ({tag}).")
            suffix = "" if tag == "cu121" else f"+{tag}"   # PyTorch's default PyPI-style tag omits +cu121
            return [sys.executable, "-m", "pip", "install", f"torch=={TORCH_VERSION}{suffix}",
                    "--index-url", index_url]

    print(f"Driver reports CUDA {major}.{minor}, which is older than any supported bucket "
          f"-> falling back to CPU-only PyTorch.")
    return [sys.executable, "-m", "pip", "install", f"torch=={TORCH_VERSION}+cpu",
            "--index-url", "https://download.pytorch.org/whl/cpu"]


def main() -> None:
    check_only = "--check" in sys.argv
    cuda_ver = detect_driver_cuda_version()
    cmd = pick_install_command(cuda_ver)

    print("Command:", " ".join(cmd))
    if check_only:
        print("(--check passed: not installing)")
        return

    # Uninstall first: a stale CPU or mismatched-CUDA build can otherwise survive because
    # pip's version comparator does not always treat local-version suffixes (+cpu/+cu121)
    # as different enough to force a clean reinstall.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torch", "torchvision", "torchaudio"], check=False)

    # These wheels are ~2.5 GB. On a connection that drops mid-download, pip's own
    # "resume incomplete download" logic can stitch together a corrupted file (the bytes
    # won't match the index's sha256, and pip correctly refuses to install it). Purging the
    # cache first prevents resuming from an already-corrupted partial; --no-cache-dir stops
    # this attempt from being cached either; a larger --retries/--timeout budget lets pip
    # restart cleanly instead of resuming.
    subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
    full_cmd = cmd + ["--no-cache-dir", "--retries", "10", "--timeout", "120"]

    for attempt in range(1, 4):
        result = subprocess.run(full_cmd)
        if result.returncode == 0:
            break
        print(f"Install attempt {attempt}/3 failed (likely a corrupted/interrupted download of "
              f"a large wheel) -- purging cache and retrying...")
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
    else:
        print("All 3 install attempts failed. If this keeps happening, your connection may be "
              "too unstable for pip's downloader -- download the wheel manually with a resumable "
              "tool (e.g. `curl -C -`) from the URL pip printed, verify its sha256 against the "
              "'Expected sha256' pip reported, then `pip install <local_wheel_path>`.")
        sys.exit(1)

    # Verify
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(torch.__version__); print('cuda_available:', torch.cuda.is_available())"],
        capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()