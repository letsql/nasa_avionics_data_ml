#! /bin/env bash
set -eux

python=${1:-python3.10}
venv_dir=${2:-./venv}

"$python" -m venv --without-pip "$venv_dir"
source "$venv_dir"/bin/activate
"$python" <(wget -O - https://bootstrap.pypa.io/get-pip.py)
