"""
This file is part of the PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from ...pypicongpu import species

import typing
import pydantic

"""
note on rms_velocity:
---------------------
The rms_velocity is converted to a temperature in keV. This conversion requires the mass of the species to be known,
which is not the case inside the picmi density distribution.

As an abstraction, **every** PICMI density distribution implements `picongpu_get_rms_velocity_si()` which returns a
tuple (float, float, float) with the rms_velocity per axis in SI units (m/s).

In case the density profile does not have an rms_velocity, this method **MUST** return (0, 0, 0), which is translated to
"no temperature initialization" by the owning species.

note on drift:
--------------
The drift ("velocity") is represented using either directed_velocity or centroid_velocity (v, gamma*v respectively) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""


class Distribution(pydantic.BaseModel):
    rms_velocity: typing.Tuple[float, float, float] = (0, 0, 0)
    """thermal velocity spread [m/s]"""

    directed_velocity: typing.Tuple[float, float, float] = (0, 0, 0)
    """Directed, average, proper velocity [m/s]"""

    fill_in: bool = True
    """Flags whether to fill in the empty spaced opened up when the grid moves"""

    def __hash__(self):
        """custom hash function for indexing in dicts"""
        hash_value = hash(type(self))
        for value in self.__dict__.values():
            hash_value += hash(value)
        return hash_value

    def picongpu_get_rms_velocity_si(self) -> typing.Tuple[float, float, float]:
        return tuple(self.rms_velocity)

    def get_picongpu_drift(self) -> typing.Optional[species.operation.momentum.Drift]:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        if [0, 0, 0] == self.directed_velocity or (0, 0, 0) == self.directed_velocity:
            return None

        drift = species.operation.momentum.Drift()
        drift.fill_from_velocity(tuple(self.directed_velocity))
        return drift
