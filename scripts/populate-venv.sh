#! /bin/env bash
set -eux

venv_dir=${1:-./venv}

source "$venv_dir"/bin/activate
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pip install poetry
poetry install
