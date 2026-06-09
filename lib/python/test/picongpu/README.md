# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# PIConGPU Python Test Suite

This directory contains the test suite for the Python bindings of PIConGPU.

## Test Categories

### Quick Tests (`quick/`)
Fast unit tests that run in seconds. Used for CI on every commit.

**Run:** `pytest quick/`

### Compiling Tests (`compiling/`)
Tests that compile PIConGPU simulations. Long-running, requires build tools (cmake, boost, C++ compiler).

**Run:** `pytest -m compiling`
**Mark:** `@pytest.mark.slow`, `@pytest.mark.compiling`

### End-to-End Tests (`end_to_end/`)
Full simulation tests comparing output against reference data.

**Run:** `pytest -m end_to_end`
**Mark:** `@pytest.mark.slow`, `@pytest.mark.end_to_end`

## Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run quick tests only (default for CI)
pytest quick/
pytest -m "not slow"  # equivalent

# Run all tests including slow ones
pytest

# Run only slow tests
pytest -m slow

# Run specific category
pytest -m compiling
pytest -m end_to_end
```

## Test Markers

- `slow` - long-running tests (compiling, end-to-end)
- `compiling` - tests that compile PIConGPU simulations
- `end_to_end` - full simulation tests with reference comparison

Markers are automatically applied based on test directory location via `conftest.py`.
