"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory

import json
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


def test_validate_rocrate(setup_dir):
    assert not validate(settings={"rocrate_uri": setup_dir, "requirement_severity": "REQUIRED"}).get_issues()


@mark.xfail(
    reason="rocrate_validator does not yet handle v1.2. "
    "When it does, we'll get informed by this test passing. "
    "And by passing I mean failing... well, you know what I mean."
)
def test_validation_of_rocrates_v_1_2():
    with TemporaryDirectory() as d:
        with (Path(d) / "ro-crate-metadata.json").open("w") as file:
            json.dump(
                # The minimal example from RO-Crate v1.2:
                # https://www.researchobject.org/ro-crate/specification/1.2/introduction.html
                {
                    "@context": "https://w3id.org/ro/crate/1.2/context",
                    "@graph": [
                        {
                            "@id": "ro-crate-metadata.json",
                            "@type": "CreativeWork",
                            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.2"},
                            "about": {"@id": "./"},
                        },
                        {
                            "@id": "./",
                            "@type": "Dataset",
                            "name": "Example dataset for RO-Crate specification",
                            "description": "Official rainfall readings for Katoomba, NSW 2022, Australia",
                            "datePublished": "2022-12-01",
                            "publisher": {"@id": "https://ror.org/04dkp1p98"},
                            "license": {"@id": "http://spdx.org/licenses/CC0-1.0"},
                            "hasPart": [{"@id": "data.csv"}],
                        },
                        {
                            "@id": "data.csv",
                            "@type": "File",
                            "name": "Rainfall data for Katoomba, NSW Australia February 2022",
                            "encodingFormat": "text/csv",
                            "license": {"@id": "https://creativecommons.org/licenses/by-nc-sa/3.0/au/"},
                        },
                        {
                            "@id": "https://ror.org/04dkp1p98",
                            "@type": "Organization",
                            "name": "Bureau of Meteorology",
                            "description": "Australian Government Bureau of Meteorology",
                            "url": "http://www.bom.gov.au/",
                        },
                        {
                            "@id": "https://creativecommons.org/licenses/by-nc-sa/3.0/au/",
                            "@type": "CreativeWork",
                            "name": "CC BY-NC-SA 3.0 AU",
                            "description": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Australia",
                        },
                    ],
                },
                file,
            )
        (Path(d) / "data.csv").touch()

        assert not validate(settings={"rocrate_uri": d}).get_issues()


def test_validation_of_rocrates_v_1_1():
    with TemporaryDirectory() as d:
        with (Path(d) / "ro-crate-metadata.json").open("w") as file:
            json.dump(
                # The minimal example from RO-Crate v1.1:
                # https://www.researchobject.org/ro-crate/specification/1.1/root-data-entity.html
                {
                    "@context": "https://w3id.org/ro/crate/1.1/context",
                    "@graph": [
                        {
                            "@type": "CreativeWork",
                            "@id": "ro-crate-metadata.json",
                            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                            "about": {"@id": "./"},
                        },
                        {
                            "@id": "./",
                            "identifier": "https://doi.org/10.4225/59/59672c09f4a4b",
                            "@type": "Dataset",
                            "datePublished": "2017",
                            "name": "Data files associated with the manuscript:Effects of facilitated family case conferencing for ...",
                            "description": "Palliative care planning for nursing home residents with advanced dementia ...",
                            "license": {"@id": "https://creativecommons.org/licenses/by-nc-sa/3.0/au/"},
                        },
                        {
                            "@id": "https://creativecommons.org/licenses/by-nc-sa/3.0/au/",
                            "@type": "CreativeWork",
                            "description": "This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Australia License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/au/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.",
                            "identifier": "https://creativecommons.org/licenses/by-nc-sa/3.0/au/",
                            "name": "Attribution-NonCommercial-ShareAlike 3.0 Australia (CC BY-NC-SA 3.0 AU)",
                        },
                    ],
                },
                file,
            )
        (Path(d) / "data.csv").touch()

        assert not validate(settings={"rocrate_uri": d}).get_issues()
