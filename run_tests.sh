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

# Run tests with pytest and coverage.
python3 -m coverage run -m pytest -v tests/
python3 -m coverage report -m
echo "All tests passed!"

popd
