#! /bin/env bash
venv_dir=${1:-./venv}

if [ ! -d "$venv_dir" ]; then
	bash scripts/make-venv.sh
	bash scripts/populate-venv.sh
fi
