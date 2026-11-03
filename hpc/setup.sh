#!/usr/bin/env bash
# One-time setup on an NYU HPC cloud-burst node: verify the private training export,
# unpack it next to the official CSVs, and build the pinned training environment.
set -euo pipefail
K=/scratch/$USER/knicks
cd "$K"

echo "ae5a9edea71063c1e4e8a06127f41cc143ced82c120ba0012fddc7bd309de5d4  official-training.tar.gz" | sha256sum -c
mkdir -p data
tar -xzf official-training.tar.gz -C data

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch==2.11.0 torchvision==0.26.0 numpy==2.1.3 pillow==13.0.0

.venv/bin/python - <<'EOF'
import hashlib, json
from pathlib import Path
root = Path("data")
expected = json.loads((root / "training-image-sha256.json").read_text())
bad = sum(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in expected.items())
print(f"verified {len(expected) - bad}/{len(expected)} training image hashes")
assert bad == 0 and len(expected) == 50_000
import torch, torchvision, numpy, PIL
print(torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__, torch.version.cuda, torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
EOF
