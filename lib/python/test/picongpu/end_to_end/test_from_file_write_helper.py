"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Standalone regression test for ``_write_reference_file`` (used by
``test_from_file.py``).

openpmd-api's C backend rejects a chunk whose memory layout is not C-contiguous
(``IndexError: strides in chunk are inefficient, not implemented!``).
``_reference_particles`` returns C-order 2-D arrays, so the per-axis column
views (``[:, i]``) handed to ``store_chunk`` are strided and would trigger that
error before any simulation step can run. This test drives the helper in
isolation (no ``picongpu`` binary required) and asserts it completes and
produces a structurally valid openPMD file.

Value-level equivalence of the written particle data is exercised by the
end-to-end test (``test_from_file.py``) through PIConGPU's own openPMD reader,
which is the authoritative interpreter of the file; it is intentionally not
re-asserted here because a plain Python openpmd-api read-back is backend- and
version-dependent.
"""

import openpmd_api as opmd
import pytest

from . import test_from_file as tff


@pytest.mark.parametrize("n_particles", [1, 8, 32])
def test_write_reference_file(tmp_path, monkeypatch, n_particles):
    monkeypatch.setattr(tff, "NUMBER_OF_PARTICLES", n_particles)

    path = tmp_path / "test.bp5"
    reference = tff._write_reference_file(path)

    # The helper must have completed (the regression under test was a hard
    # ``IndexError`` at ``store_chunk``) and emitted a non-empty file.
    assert path.exists() and path.stat().st_size > 0
    assert all(arr.shape[0] == n_particles for arr in reference.values())

    # Structural sanity of the written file: the expected component tree,
    # per-axis dataset sizes, and the unit-SI metadata.
    series = opmd.Series(str(path), opmd.Access.read_only)
    try:
        particles = series.iterations[0].particles[tff.SPECIES_NAME]
        for record in ("position", "positionOffset", "momentum"):
            for i, axis in enumerate(("x", "y", "z")):
                comp = particles[record][axis]
                assert list(comp.shape) == [n_particles]
                expect_unit = tff.CELL_SIZE[i] if record == "position" else 1.0
                assert comp.unit_SI == expect_unit
        scalar = opmd.Mesh_Record_Component.SCALAR
        weighting = particles["weighting"][scalar]
        assert list(weighting.shape) == [n_particles]
        assert weighting.unit_SI == 1.0
    finally:
        series.close()
