#!/bin/sh
set -eu
python -m pip install uv
uv pip install --system dist/*.whl
uv pip install --system --group bindings/python/pyproject.toml:free-threaded-tests
python -m pytest -q bindings/python/tests/test_threadable.py::test_free_threaded_roundtrip bindings/python/tests/test_multithreaded.py::test_multithreaded_roundtripping_numpy
