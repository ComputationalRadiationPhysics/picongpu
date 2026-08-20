"""Tests for rendering built-in openPMD derived fields."""

from picongpu import templates
from picongpu.pypicongpu.output.openpmd_plugin import BuiltinFieldSolver, FieldDump
from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDConfig, OpenPMDPlugin
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor import ParticleFunctor
from picongpu.pypicongpu.rendering import Renderer
from picongpu.pypicongpu.simulation import Simulation


def test_builtin_solver_rendering_has_no_functor_struct():
    field = FieldDump(
        name="electrons_all_weightedVelocity/x",
        species="species_electrons",
        filtername=None,
        builtin_solver=BuiltinFieldSolver(
            type="deriveField::derivedAttributes::WeightedVelocity<0>",
            typename="Native_WeightedVelocity_0",
        ),
    )
    template = (templates.path() / "include/picongpu/param/fileOutput.param.mustache").read_text()
    context = {
        "derived_field_functors": [],
        "field_tmp_solvers": [field.get_solver().model_dump(mode="json")],
    }

    rendered = Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), template)

    assert "struct" not in rendered
    assert "Native_WeightedVelocity_0_species_electrons_All_Seq" not in rendered
    assert "deriveField::derivedAttributes::WeightedVelocity<0>" in rendered
    assert "species_electrons" in rendered
    assert "deriveField::CreateFieldTmpOperation_t<" in rendered
    assert "CreateEligible_t" not in rendered
    assert "VectorAllSpecies" not in rendered.split("using FieldTmpSolvers", maxsplit=1)[0]
    assert "static_assert" in rendered


def test_solver_deduplication_includes_species_and_filter(tmp_path):
    solver = BuiltinFieldSolver(type="deriveField::derivedAttributes::Density", typename="Native_Density")
    period = TimeStepSpec(specs=[{"start": 0, "stop": 10, "step": 1}])
    plugin = OpenPMDPlugin(
        sources=[
            (
                period,
                FieldDump(
                    name="electrons_all_density",
                    species="species_electrons",
                    filtername=None,
                    builtin_solver=solver,
                ),
            ),
            (
                period,
                FieldDump(
                    name="electrons_all_density_duplicate",
                    species="species_electrons",
                    filtername=None,
                    builtin_solver=solver,
                ),
            ),
            (
                period,
                FieldDump(
                    name="ions_all_density",
                    species="species_ions",
                    filtername=None,
                    builtin_solver=solver,
                ),
            ),
            (
                period,
                FieldDump(
                    name="electrons_filtered_density",
                    species="species_electrons",
                    filtername="EnergyFilter",
                    builtin_solver=solver,
                ),
            ),
        ],
        config=OpenPMDConfig(file="simData"),
    )
    plugin.setup_dir = tmp_path
    (tmp_path / "etc").mkdir()

    simulation = Simulation.model_construct(output=[plugin])
    solvers = simulation.field_tmp_solvers

    assert len(solvers) == 3
    assert {entry.species for entry in solvers} == {"species_electrons", "species_ions"}
    assert {entry.filtername for entry in solvers} == {None, "EnergyFilter"}

    serialized_plugin = plugin.model_dump(mode="json")
    assert len(serialized_plugin["sources"]) == 4
    assert serialized_plugin["sources"][0]["source"]["name"] == "electrons_all_density"


def test_custom_functor_definition_is_reused_by_species_specific_solvers():
    functor = ParticleFunctor(
        name="custom_density",
        functor_expression="1.0",
        functor_preamble=[],
        return_type=float,
    )
    period = TimeStepSpec(specs=[{"start": 0, "stop": 10, "step": 1}])
    plugins = [
        OpenPMDPlugin(
            sources=[
                (
                    period,
                    FieldDump(
                        name=f"{species}_all_custom_density",
                        species=f"species_{species}",
                        filtername=None,
                        functor=functor,
                    ),
                )
            ]
        )
        for species in ("electrons", "ions")
    ]
    simulation = Simulation.model_construct(output=plugins)

    assert len(simulation.derived_field_functors) == 1
    assert len(simulation.field_tmp_solvers) == 2
    assert simulation.field_tmp_solvers[0].attribute_type == simulation.field_tmp_solvers[1].attribute_type
