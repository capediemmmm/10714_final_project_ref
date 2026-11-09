PYTHON ?= python3
.PHONY: all lib test
all: lib
lib:
	@cmake -S . -B build -DPython_EXECUTABLE="$$(command -v $(PYTHON))" -DCMAKE_BUILD_TYPE=Release
	@cmake --build build -j 4

test: lib
	@PYTHONPATH=python:. $(PYTHON) -m pytest -q tests
