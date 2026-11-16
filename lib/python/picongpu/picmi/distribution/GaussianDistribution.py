"""
This file is part of PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.picmi._copy_attributes import converts_to
from picongpu.pypicongpu.species.operation.densityprofile.gaussian import Gaussian
from ...pypicongpu import util

from .Distribution import Distribution

import numpy as np


# Sentinel to distinguish an omitted ``z`` from an explicit one.
_MISSING_Z = object()


@converts_to(
    Gaussian,
    conversions={"vacuum_cells_front": lambda self, _: int(self.vacuum_front / self.cell_size[1])},
    preamble=lambda self, grid: setattr(self, "cell_size", grid.picongpu_cell_size) or self.check(),
    ignore=["check"],
)
class GaussianDistribution(Distribution):
    r"""
    Describes a density distribution of particles with gaussian up- and down-ramps with a constant density region in
    between

    Will create the following profile:

    * for :math:`y < c_\mathrm{front}`:
      :math:`n(y) = \rho \exp\left(f \left|\frac{y - c_\mathrm{front}}{\sigma_\mathrm{front}}\right|^p\right)`
    * for :math:`c_\mathrm{front} \le y \le c_\mathrm{rear}`:
      :math:`n(y) = \rho`
    * for :math:`y > c_\mathrm{rear}`:
      :math:`n(y) = \rho \exp\left(f \left|\frac{y - c_\mathrm{rear}}{\sigma_\mathrm{rear}}\right|^p\right)`

    where :math:`n` is the density, :math:`\rho` is ``density``,
    :math:`f` is ``factor``, :math:`p` is ``power``, and :math:`y` is the
    position in the simulation box

    The profile is one-dimensional along ``y``. It can be evaluated on both 3D
    and 2D grids, with a dimension-dependent call contract (see :meth:`__call__`).
    """

    density: float
    """particle number density, [m^-3]"""

    center_front: float
    """center of gaussian ramp at the front, [m]"""
    center_rear: float
    """center of the gaussian ramp at the rear, [m]"""

    sigma_front: float
    """sigma of the gaussian ramp at the front, [m]"""
    sigma_rear: float
    """sigma of the gaussian ramp at the rear, [m]"""

    power: float
    """power used in exponential function, 2 will yield a gaussian, 4+ a super-gaussian, unitless"""
    factor: float
    """sign and scaling factor, must be < 0, unitless"""

    vacuum_front: float
    """size of the vacuum in front of density, gets rounded down to full cells, [m]"""

    lower_bound: tuple[float, float, float] | tuple[None, None, None] = (
        None,
        None,
        None,
    )
    upper_bound: tuple[float, float, float] | tuple[None, None, None] = (
        None,
        None,
        None,
    )

    cell_size: tuple[float, float] | tuple[float, float, float] | None = None
    """cell size per spatial dimension; a 2-tuple on 2D grids, a 3-tuple on 3D grids"""

    # @details pydantic provides an automatically generated __init__/constructor method which allows initialization off
    #   all attributes as keyword arguments

    def check(self):
        util.unsupported("fill in not active", self.fill_in, True)
        util.unsupported("lower bound", self.lower_bound, (None, None, None))
        util.unsupported("upper bound", self.upper_bound, (None, None, None))
        if self.center_rear < self.center_front:
            raise ValueError("center_front must be <= center_rear")
        if self.density <= 0.0:
            raise ValueError("density must be > 0")

    def __call__(self, x, y, z=_MISSING_Z):
        """
        Evaluate the density.

        The call contract follows the grid dimensionality (stored in
        ``cell_size``):

        * 3D grid: ``__call__(x, y, z)`` -- the usual three spatial coordinates.
        * 2D grid: ``__call__(x, y)`` -- ``z`` does not exist and passing it is an
          error. The profile itself is still defined along ``y``.
        """
        if self.cell_size is None:
            message = (
                "Due to inconsistencies in the backend, evaluation of this function requires information about the cell_size."
                " You can either set it manually "
                " or you can perform anything that includes writing the input files on your simulation object."
                " This is a temporary workaround and will be fixed in the future."
            )
            raise NotImplementedError(message)

        is_2d = len(self.cell_size) == 2
        if is_2d:
            if z is not _MISSING_Z:
                raise TypeError(
                    "GaussianDistribution on a 2D grid is evaluated as __call__(x, y); the z coordinate does not exist."
                )
            coordinates = (x, y)
        else:
            if z is _MISSING_Z:
                raise TypeError("GaussianDistribution on a 3D grid is evaluated as __call__(x, y, z).")
            coordinates = (x, y, z)

        # The definition of this density uses the origin of the cell
        # while the call operator uses the center.
        shifted = [np.asarray(c) - 0.5 * cell_size for c, cell_size in zip(coordinates, self.cell_size)]
        x, y = shifted[:2]

        # The last term undoes the shift to the cell origin.
        vacuum_y = int(self.vacuum_front / self.cell_size[1]) * self.cell_size[1] - 0.5 * self.cell_size[1]

        # We do this to get the correct shape after broadcasting:
        exponent = np.zeros(np.broadcast_shapes(*[np.shape(coordinate) for coordinate in shifted]))
        exponent[y < self.center_front] = np.abs((y - self.center_front) / self.sigma_front)[y < self.center_front]
        exponent[y >= self.center_rear] = np.abs((y - self.center_rear) / self.sigma_rear)[y >= self.center_rear]

        result = np.asarray(np.exp(self.factor * exponent**self.power))
        result[y < vacuum_y] = 0.0
        return self.density * result
