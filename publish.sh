#!/bin/bash
set -o errexit

# This script requires these tools:
# pip3 install --user --upgrade setuptools wheel build twine

PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "${PROJECT_PATH}"

# Clean up previous builds
rm -rf build
rm -rf dist
rm -rf *.egg-info

# Build package (sdist and wheel)
python3 setup.py sdist bdist_wheel

# Validate package metadata and upload to PyPI
python3 -m twine check dist/*
if [[ "$1" != "--build-only" ]]; then
    python3 -m twine upload dist/*
fi

popd
