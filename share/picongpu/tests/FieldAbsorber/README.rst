.. _tests-fieldAbsorber:

Current Source Radiating in an Unbounded Three-Dimensional Region
=================================================================

.. sectionauthor:: Klaus Steiniger, Sergey Ermakov
.. moduleauthor:: Sergei Bastrakov, Klaus Steiniger

Example of a rectangular conductor with a steady current.

This setup is based on [Taflove2005]_, section 7.11.1.
The difference is we consider both 2D and 3D cases, and grid size may be increased due to our absorber being part of the simulation area, not located outside of it as in the book.

The conductor is located within the center of the x-y-plane, oriented along the z-axis and infinitely extended along this axis, but current flows along y.
The conductor properties can be adjusted in `include/picongpu/param/fieldBackground.param` within the `class FieldBackgroundJ`.
Specifically, the variable `halfWidth` adjusts its edge length and can be used to apply the test with meaningful results to higher-order solvers, too.
The current in the wire ramps up over time according to a differentiated Gaussian.
This defines the current density amplitude, too.
Therefore, the total current through the wire scales with the wire's halfWidth.

The simulation is performed twice.
First the actual PML test is performed with a small simulation volume, here 60x60x60 cells.
The simulation runs for 600 steps which is much longer than the steady state response.
Second, a reference simulation is performed with a large simulation volume of 660x660x660 cells.
PML boundaries are applied at the boundaries with normals along the x and y axes.
Periodic boundaries are applied at the boundary whose normal points along the z axis.

Each component of the electric field is probed over time at two different positions, each of which has the same distance from the conductor in the two simulations.
That is, these points have to be less than 30 cells away from the conductor.

Since the reference grid is sufficiently large to avoid reflections from the boundary at the points of measurement, the PML quality is defined as the relative error

.. math::

  \mathrm{Rel. error}\rvert^{n}_{i,j} =
    \frac{\left| E\rvert^{n}_{i,j} - E_\mathrm{ref}\rvert^{n}_{i,j}  \right|}{\left| E_\mathrm{ref,max}\rvert_{i,j}  \right|}

with :math:`n` being the time step, :math:`(i,j)` the distance in cells from the conductor, :math:`E_\mathrm{ref,max}\rvert_{i,j}` the maximum amplitude of the reference field at :math:`(i,j)` over the total simulation duration.

According to [Taflove2005]_, the PML quality should be of the order of :math:`10^{-5}` for a 10 cell CPML.

References
----------

.. [Taflove2005]
        A. Taflove, S.C. Hagness
        *Computational electrodynamics: the finite-difference time-domain method*
        Artech house (2005)
