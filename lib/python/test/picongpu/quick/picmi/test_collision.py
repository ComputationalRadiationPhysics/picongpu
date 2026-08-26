"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonschema
import pytest
from picongpu import picmi
from picongpu.picmi.interaction.collision import (
    Collision,
    CollisionalPhysicsSetup,
    ConstLogCollision,
    DynamicLogCollision,
)
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies as PicmiFilteredSpecies
from picongpu.picmi.particle_functor.particle_filter import ParticleFilter
from picongpu.pypicongpu.collisions import Collision as PyPIConGPUCollision
from picongpu.pypicongpu.collisions import SpeciesPair

COULOMB_LOG = 0.0920


@pytest.fixture
def grid():
    number_of_cells = 32
    cell_size = 1
    return picmi.Cartesian3DGrid(
        number_of_cells=[number_of_cells] * 3,
        lower_bound=[0, 0, 0],
        upper_bound=list(map(lambda x: number_of_cells * x, [cell_size] * 3)),
        # required, otherwise won't spawn
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )


@pytest.fixture
def electrons():
    return picmi.Species(name="electron", particle_type="electron")


@pytest.fixture
def hydrogen():
    return picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)


def get_collision_param(setup_dir: Path) -> str:
    return (setup_dir / "include/picongpu/param/collision.param").read_text()


def assert_coulomb_log(rendered_param: str, expected: float) -> None:
    match = re.search(r"static constexpr float_X coulombLog = ([^_]+)_X;", rendered_param)
    assert match, f"no coulombLog found in rendered collision.param:\n{rendered_param}"
    assert abs(float(match.group(1)) - expected) < 1e-12


def test_collision_functor_serialization():
    # regression #5754: the functor is serialized with its natural flat shape
    # (plain model dump), which is exactly what the mustache template reads.
    assert ConstLogCollision(coulomb_log=COULOMB_LOG).model_dump(mode="json") == {
        "type_constlog": True,
        "coulomb_log": COULOMB_LOG,
    }
    assert DynamicLogCollision().model_dump(mode="json") == {"type_dynamiclog": True}


def assert_collision_dump_validates(pypicongpu_collision: PyPIConGPUCollision) -> None:
    # regression #5754: the JSON schema must match the serializer output
    # (one {species_lhs, species_rhs} dict per pair, flat functor context).
    context = pypicongpu_collision.model_dump(mode="json")
    schema = PyPIConGPUCollision.model_json_schema(mode="serialization")
    jsonschema.Draft202012Validator(schema).validate(context)

    for pair in context["species_pairs"]:
        assert set(pair) == {"species_lhs", "species_rhs"}
        for species in pair.values():
            assert species["typename"]
    if isinstance(pypicongpu_collision.functor, ConstLogCollision):
        assert context["functor"] == {
            "type_constlog": True,
            "coulomb_log": pypicongpu_collision.functor.coulomb_log,
        }


def test_collision_dump_validates_against_schema(grid, electrons, hydrogen):
    sim = picmi.Simulation(
        time_step_size=17,
        max_steps=4,
        solver=picmi.ElectromagneticSolver(method="Yee", grid=grid),
    )
    sim.add_species(electrons, None)
    sim.add_species(hydrogen, None)
    sim.picongpu_interaction = [
        Collision.construct_from_pairs(
            species_pairs=[(electrons, hydrogen)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG)
        )
    ]
    pypicongpu_collision = sim.get_as_pypicongpu().collisional_physics.collisions[0]
    assert_collision_dump_validates(pypicongpu_collision)


def test_write_input_file_with_const_log_collision(grid, electrons, hydrogen):
    # regression #5754: write_input_file() must complete for a ConstLogCollision
    # and generate a valid collision.param.
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    sim = picmi.Simulation(
        time_step_size=17,
        max_steps=4,
        solver=solver,
        picongpu_interaction=[
            Collision.construct_from_pairs(
                species_pairs=[(electrons, hydrogen)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG)
            )
        ],
    )
    sim.add_species(electrons, None)
    sim.add_species(hydrogen, None)

    # the rendering context must pass the (auto-generated) JSON schema check
    pypicongpu_sim = sim.get_as_pypicongpu()
    assert pypicongpu_sim.get_rendering_context() != {}
    assert_collision_dump_validates(pypicongpu_sim.collisional_physics.collisions[0])

    with TemporaryDirectory() as tmpdir:
        setup_dir = str(Path(tmpdir) / "generated-input")
        # first generation into a not-yet existing directory
        sim.write_input_file(setup_dir)
        assert_coulomb_log(get_collision_param(Path(setup_dir)), COULOMB_LOG)

        # second generation into the already existing directory
        sim.write_input_file(setup_dir, exist_ok=True)
        assert_coulomb_log(get_collision_param(Path(setup_dir)), COULOMB_LOG)


def test_write_input_file_with_mixed_collision_functors(grid, electrons, hydrogen):
    # the other functor (dynamic log) must keep working next to the const log
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    setup = CollisionalPhysicsSetup(
        collisions=[
            Collision.construct_from_pairs(
                species_pairs=[(electrons, hydrogen)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG)
            ),
            Collision.construct_from_pairs(species_pairs=[(electrons, hydrogen)], functor=DynamicLogCollision()),
        ],
        screening_species=[electrons],
    )
    sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_interaction=[setup])
    sim.add_species(electrons, None)
    sim.add_species(hydrogen, None)

    with TemporaryDirectory() as tmpdir:
        sim.write_input_file(str(Path(tmpdir) / "generated-input"), exist_ok=True)
        rendered_param = get_collision_param(Path(tmpdir) / "generated-input")
        assert "RelativisticCollisionDynamicLog" in rendered_param
        assert_coulomb_log(rendered_param, COULOMB_LOG)
        assert "Pair<species_electron,species_hydrogen>" in rendered_param


def test_bare_collisions_are_merged_into_a_setup(grid, electrons, hydrogen):
    # regression #5754: a list of bare collisions (with no other interaction)
    # must not be rejected by the constructor and must not be dropped
    # by the conversion to pypicongpu.
    sim = picmi.Simulation(
        time_step_size=17,
        max_steps=4,
        solver=picmi.ElectromagneticSolver(method="Yee", grid=grid),
        picongpu_interaction=[
            Collision.construct_from_pairs(
                species_pairs=[(electrons, hydrogen)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG)
            )
        ],
    )
    sim.add_species(electrons, None)
    sim.add_species(hydrogen, None)

    collisions = sim.get_as_pypicongpu().collisional_physics.collisions
    assert len(collisions) == 1
    assert collisions[0].functor == ConstLogCollision(coulomb_log=COULOMB_LOG)


def test_species_pair_from_tuple_or_model(electrons, hydrogen):
    # both the historical bare 2-tuple form and an explicit SpeciesPair must be accepted
    lhs, rhs = electrons.get_as_pypicongpu(), hydrogen.get_as_pypicongpu()
    from_tuple = PyPIConGPUCollision(species_pairs=[(lhs, rhs)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG))
    from_model = PyPIConGPUCollision(
        species_pairs=[SpeciesPair(species_lhs=lhs, species_rhs=rhs)],
        functor=ConstLogCollision(coulomb_log=COULOMB_LOG),
    )
    assert from_tuple.species_pairs == from_model.species_pairs
    assert [s.name for s in from_tuple.species] == ["electron", "hydrogen"]


def test_intra_species_collision_filters(electrons):
    # the same filter on both sides is fine...
    shared = ParticleFilter(functor=lambda _: True, name="same_filter")
    same_a = PicmiFilteredSpecies(species=electrons, functor=shared).get_as_pypicongpu()
    same_b = PicmiFilteredSpecies(species=electrons, functor=shared).get_as_pypicongpu()
    assert PyPIConGPUCollision(species_pairs=[(same_a, same_b)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG))
    # ...but differently filtered species of the same species are rejected
    different_a = PicmiFilteredSpecies(
        species=electrons, functor=ParticleFilter(functor=lambda _: True, name="filter_a")
    ).get_as_pypicongpu()
    different_b = PicmiFilteredSpecies(
        species=electrons, functor=ParticleFilter(functor=lambda _: True, name="filter_b")
    ).get_as_pypicongpu()
    with pytest.raises(ValueError):
        PyPIConGPUCollision(
            species_pairs=[(different_a, different_b)], functor=ConstLogCollision(coulomb_log=COULOMB_LOG)
        )
