#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")" && pwd)
python "$repo_dir/visualize_density_components_ase.py" "$@"
