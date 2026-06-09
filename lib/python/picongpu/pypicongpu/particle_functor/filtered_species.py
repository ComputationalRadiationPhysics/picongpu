# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, computed_field

from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.rendering.renderedobject import RenderedObject
from picongpu.pypicongpu.species.species import Species


class FilteredSpecies(BaseModel, RenderedObject):
    species: Species
    functor: ParticleFunctor

    @computed_field
    def name_with_filter(self) -> str:
        return f"{self.species.name}_{self.functor.name}"

    @computed_field
    def species_name(self) -> str:
        return self.species.name

    @computed_field
    def filter_name(self) -> str:
        return self.functor.name

    @computed_field
    def filter_typename(self) -> str:
        return self.filter_name

    @computed_field
    def typename(self) -> str:
        return self.species.typename

    @computed_field
    def name(self) -> str:
        return self.name_with_filter

    @computed_field
    def type_filtered(self) -> bool:
        return True
