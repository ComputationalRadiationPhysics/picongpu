"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory

from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation
from picongpu._version import __version__
from pytest import fixture, mark
from rocrate.rocrate import ROCrate
from rocrate_validator.services import validate


@fixture
def sim():
    number_of_cells = 32
    cell_size = 1
    return Simulation(
        time_step_size=17,
        max_steps=4,
        solver=ElectromagneticSolver(
            method="Yee",
            grid=Cartesian3DGrid(
                number_of_cells=[number_of_cells, number_of_cells, number_of_cells],
                lower_bound=[0, 0, 0],
                upper_bound=list(map(lambda x: number_of_cells * x, [cell_size, cell_size, cell_size])),
                # required, otherwise won't spawn
                lower_boundary_conditions=["open", "open", "periodic"],
                upper_boundary_conditions=["open", "open", "periodic"],
            ),
        ),
    )


@fixture
def setup_dir(sim):
    with TemporaryDirectory() as d:
        sim.write_input_file(d, exist_ok=True)
        yield Path(d)


@fixture
def crate(setup_dir):
    return ROCrate(setup_dir)


def test_all_files_tracked_by_rocrate(setup_dir, crate):
    tracked = [Path(setup_dir) / e.id for e in crate.data_entities if "File" in e.type]
    existing = list(filter(Path.is_file, Path(setup_dir).rglob("*")))
    # The `ro-crate-metadata.json` are listed as a different @type.
    assert set(existing).symmetric_difference(tracked) == {Path(setup_dir) / "ro-crate-metadata.json"}


@mark.xfail(
    reason="Undecided whether we should set it up like this. "
    "I'm just temporarily leaving it as a reminder "
    "and to spare the trouble to write the test again if we decide positively."
)
def test_all_directories_record_their_content_as_has_part(setup_dir, crate):
    for id_, parent_id in map(lambda p: _as_ids(p, setup_dir), setup_dir.rglob("*")):
        if id_ != "ro-crate-metadata.json":
            entity = crate.get(parent_id)
            assert entity is not None
            assert id_ in map(itemgetter("@id"), entity.properties()["hasPart"])
    for entity in crate.get_entities():
        for part in entity.properties.get("hasPart", []):
            assert (Path(setup_dir) / part["@id"]).exists()


def _as_ids(p: Path, relative_to: Path):
    local = p.relative_to(relative_to)
    return map(lambda x: str(x) + ("/" if (relative_to / x).is_dir() else ""), (local, local.parent))


def test_rocrate_has_basic_metadata(crate):
    assert crate.name is not None
    assert crate.description is not None


def test_rocrate_points_main_entity_to_workflow(crate):
    assert crate.mainEntity.properties()["@id"] == "workflow/workflow.cwl"


def test_rocrate_indicates_the_software_it_has_been_produced_with(crate):
    picongpu_doi = "https://doi.org/10.5281/zenodo.14513363"
    assert (picongpu_software := crate.get(picongpu_doi).properties()["@id"]) is not None
    assert crate.get(picongpu_doi).properties()["version"] == __version__
    assert picongpu_software in map(itemgetter("@id"), crate.root_dataset.properties()["instrument"])


@mark.xfail(reason="Not implemented yet.")
def test_adds_default_information_to_datasets(crate):
    # explicitly instantiating a list here for pytest to provide better assertion error messages
    assert all(["description" in dataset.properties() for dataset in crate.get_by_type("Dataset")])


@mark.xfail(reason="Decided to disable license until we have a proper interface.")
def test_validate_rocrate(setup_dir):
    assert not validate(settings={"rocrate_uri": setup_dir, "requirement_severity": "REQUIRED"}).get_issues()
