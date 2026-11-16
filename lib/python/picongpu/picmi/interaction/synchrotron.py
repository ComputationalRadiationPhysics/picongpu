"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel
from picongpu.picmi.species import _DependsOn, Species
from picongpu.picmi._species_requirements import SynchrotronConstantConstruction
from picongpu.pypicongpu.species.constant.synchrotron import SynchrotronParams


class Synchrotron(BaseModel):
    electron_species: Species
    photon_species: Species
    synchrotron_parameters: SynchrotronParams = SynchrotronParams()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.electron_species.register_requirements(
            [
                _DependsOn(species=self.photon_species),
                SynchrotronConstantConstruction(photon_species=self.photon_species),
            ]
        )

    def get_as_pypicongpu(self):
        return self.synchrotron_parameters
