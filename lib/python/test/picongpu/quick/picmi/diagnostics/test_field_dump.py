"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import re
from unittest import TestCase

from picongpu import templates
from picongpu.pypicongpu.output.openpmd_plugin import FieldDump
from picongpu.pypicongpu.particle_functor import ParticleFunctor
from picongpu.pypicongpu.rendering.renderer import Renderer

TEMPLATE_PATH = templates.path() / "include" / "picongpu" / "param" / "fileOutput.param.mustache"


def render_file_output(derived_fields):
    """Render the real fileOutput template with the given derived-field contexts."""
    context = {"output": [{"type_openPMD": True, "derived_fields": derived_fields}]}
    return Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), TEMPLATE_PATH.read_text())


def make_derived_field_context(species_name, functor_name="energy"):
    """Build the rendering context for one derived field dump on ``species_name``.

    Returns ``(field_context, typename)`` where ``typename`` is the emitted
    derived-attribute struct name (carries a per-functor UUID).
    """
    functor = ParticleFunctor(
        name=functor_name,
        functor_expression="Ekin",
        functor_preamble=[],
        return_type="float_64",
    )
    field = FieldDump(
        name=f"{species_name}_all_{functor_name}",
        functor=functor,
        filtername=None,
        species_name=species_name,
    )
    field_context = field.get_rendering_context()
    return field_context, field_context["functor"]["typename"]


class TestFileOutputFieldDump(TestCase):
    def test_derived_field_eligibility_specialization_is_qualified(self):
        """The emitted SpeciesEligibleForSolver specialization must fully qualify the
        derived attribute (``particleToGrid::derivedAttributes::``), not the bare
        ``derivedAttributes::`` which does not resolve from ``namespace particles::traits``.
        (Regression: the unqualified form is a hard C++ compile error.)
        """
        field_context, typename = make_derived_field_context("electron")
        rendered = render_file_output([field_context])

        qualified = f"particleToGrid::derivedAttributes::{typename}"
        self.assertIn(qualified, rendered, "emitted trait must fully qualify the derived attribute")

        # the unqualified specialization form must not appear anywhere
        unqualified = re.search(r"SpeciesEligibleForSolver<T_Species,\s*derivedAttributes::", rendered)
        self.assertIsNone(unqualified, "emitted trait must not reference a bare 'derivedAttributes::'")

    def test_derived_field_eligibility_keys_on_species_name(self):
        """The emitted specialization narrows on the species' compile-time name via
        ``GetCTName`` / ``PMACC_CSTRING``."""
        field_context, _ = make_derived_field_context("electron")
        rendered = render_file_output([field_context])

        self.assertIn('std::is_same_v<pmacc::traits::GetCTName_t<T_Species>, PMACC_CSTRING("electron")>', rendered)

    def test_native_field_dump_emits_no_eligibility_specialization(self):
        """A native E/B/J dump (functor=None) must not emit a derived attribute struct
        or a species-eligibility narrowing block."""
        native = FieldDump(name="E", functor=None, filtername=None, species_name=None).get_rendering_context()
        rendered = render_file_output([native])

        self.assertNotIn("namespace particles::traits", rendered)
        self.assertNotIn("struct SpeciesEligibleForSolver", rendered)
        self.assertNotIn("CreateEligible_t", rendered)
