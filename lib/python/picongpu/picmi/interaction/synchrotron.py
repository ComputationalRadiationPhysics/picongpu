"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from pydantic import BaseModel
from picongpu.picmi.species import DependsOn, Species
from picongpu.picmi.species_requirements import SynchrotronConstantConstruction
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams


class Synchrotron(BaseModel):
    electron_species: Species
    photon_species: Species
    synchrotron_parameters: SynchrotronParams = SynchrotronParams()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.electron_species.register_requirements(
            [
                DependsOn(species=self.photon_species),
                SynchrotronConstantConstruction(photon_species=self.photon_species),
            ]
        )

    def get_as_pypicongpu(self):
        return self.synchrotron_parameters
