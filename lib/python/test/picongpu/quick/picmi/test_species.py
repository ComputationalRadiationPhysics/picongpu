"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from unittest import TestCase

import pytest
from picongpu.picmi.interaction.ionization.fieldionization import ADK, BSI
from picongpu.picmi.species import Species
from picongpu.picmi.species_requirements import RequirementConflict, SetChargeStateOperation, run_construction
from picongpu.pypicongpu.species.attribute.weighting import Weighting
from picongpu.pypicongpu.species.constant.mass import Mass
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState


def unique_in(elements, collection):
    collection = list(collection)
    return (collection.count(e) == 1 for e in elements)


class TestSpeciesRequirementResolution(TestCase):
    def test_deduplicate_attributes(self):
        species = Species(name="dummy")
        requirements = [Weighting()]
        species.register_requirements(2 * requirements)
        assert all(unique_in(requirements, species.get_as_pypicongpu().attributes))

    def test_deduplicate_delayed_construction(self):
        species = Species(name="dummy", particle_type="H", charge_state=1)
        requirements = [SetChargeStateOperation(species)]
        species.register_requirements(2 * requirements)
        assert all(unique_in(requirements, species.get_operation_requirements()))

    def test_conflicting_constants(self):
        species = Species(name="dummy")
        requirements = [Mass(mass_si=1.0), Mass(mass_si=2.0)]
        with pytest.raises(RequirementConflict):
            # Not yet decided which one should raise, but one of them definitely will.
            species.register_requirements(requirements)
            species.get_as_pypicongpu()

    def test_ionization(self):
        ion = Species(name="ion", particle_type="H", charge_state=1)
        electron = Species(name="electron", particle_type="electron")
        # These all register requirements:
        ionizations = [
            # Not great: Production code would use the enums not their integer represenation.
            ADK(ion_species=ion, ionization_electron_species=electron, ADK_variant=0, ionization_current=None),
            BSI(ion_species=ion, ionization_electron_species=electron, BSI_extensions=[0], ionization_current=None),
        ]

        # Ionization makes the ion depend on the electron species.
        # This is important for rendering the corresponding C++ header,
        # so the electron species gets defined before the ion species.
        assert electron < ion

        set_charge_state_op = [
            run_construction(op) for op in ion.get_operation_requirements() if op.metadata.Type == SetChargeState
        ][0]
        assert set_charge_state_op.charge_state == ion.charge_state
        assert len(ion.get_as_pypicongpu().constants.ground_state_ionization.ionization_model_list) == len(ionizations)
