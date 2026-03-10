from .exponential import Exponential
from .none import None_

AllPlasmaRamps = Exponential | None_
__all__ = ["Exponential", "None_", "AllPlasmaRamps"]
