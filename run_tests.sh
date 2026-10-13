#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add project modules to PYTHONPATH.
if [[ "${PYTHONPATH}" != *"${PROJECT_PATH}"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
fi

pushd "${PROJECT_PATH}"

rm -f .coverage

if python3 -c "import pytest" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif [[ -x "/tmp/mlfont_venv/bin/python3" ]]; then
    PYTHON_BIN="/tmp/mlfont_venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

# Run tests with pytest (and coverage if installed).
if "${PYTHON_BIN}" -c "import coverage" >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m coverage run -m pytest -v tests/
    "${PYTHON_BIN}" -m coverage report -m
else
    "${PYTHON_BIN}" -m pytest -v tests/
fi
echo "All tests passed!"

popd
