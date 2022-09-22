.. _usage-workflows-boundaryConditions:

Boundary Conditions
-------------------

.. sectionauthor:: Sergei Bastrakov, Lennert Sprenger

Two kinds of boundary conditions are supported: periodic and absorbing.
They are set in a :ref:`.cfg file <usage-tbg>` with option ``--periodic <x> <y> <z>``.
Value 0 corresponds to absorbing boundaries along the axis (used by default), 1 corresponds to periodic.
The same boundary condition kind is applied for all particles species and fields, on both sides.

Particles
"""""""""

By default, boundary kinds match the value of ``--periodic`` and so are either periodic or absorbing.
For species with a particle pusher, it can be overridden with option `--<prefix>_boundary <x> <y> <z>`.
The supported boundary kinds are: periodic, absorbing, reflecting, and thermal.

Currently only the following combinations of field and particle boundaries are supported.
When fields are periodic along an axis, boundaries for all species must be periodic along this axis.
When fields are absorbing (non-periodic), species must be absorbing, reflecting, or thermal.

By default, the particle boundaries are applied at the global domain boundaries.
A user can change the boundary application area by setting an offset with the
option `--<prefix>_boundaryOffset <x> <y> <z>`.
The `boundaryOffset` is in terms of whole cells, so integers are expected.
It sets an offset inwards from the global domain boundary.
Periodic boundaries only allow 0 offset, thermal boundaries require a positive offset, and other kinds support non-negative offsets.

Boundary temperature for thermal boundaries, in keV, is set with option `--<prefix>_boundaryTemperature <x> <y> <z>`.

For example, reflecting and thermal boundary conditions for species `e` are configured by
`--e_boundary reflecting thermal reflecting`
`--e_boundaryOffset 0 1 10`
`--e_boundaryTemperature 0.0 20.0 0.0`

Particles are not allowed to be outside the boundaries for the respective species.
(For the periodic case, there are no boundaries in that sense.)
After species are initialized, all outer particles will be deleted.
During the simulation, the crossing particles will be handled by boundary condition implementations and moved or deleted as a result.

The internal treatment of particles in the guard area is controlled by the ``boundaryCondition`` flag in :ref:`speciesDefinition.param <usage-params-core>`.
However, this option is for expert users and generally should not be modified.
To set physical boundary conditions, use the command-line option described above.

Fields
""""""

Periodic boundary conditions for fields do not allow customization or variants.
The rest of the section concerns absorbing boundaries.

For the absorbing boundaries, there is a virtual field absorber layer inside the global simulation area near its boundaries.
Field values in the layer are treated differently from the rest, by a combination of a field solver and a field absorber.
It causes field dynamics inside the absorber layer to differ from that in a vacuum.
Otherwise, the affected cells are a normal part of a simulation in terms of indexing and output.
Note that particle absorption happens at the external surface of the field absorber layer, matching the global simulation area border.

The field absorber mechanism and user-controlled parameters depend on the field absorber kind enabled.
It is controlled by command-line option ``--fieldAbsorber``.
For all absorber kinds, the parameters are controlled by :ref:`fieldAbsorber.param <usage-params-core>`.

By default, the Perfectly Matched Layer (PML) absorber is used.
For this absorber, thickness of 8 to 12 cells is recommended.
Other absorber parameters can generally be used with default values.
PML generally provides much better absorber qualities than the exponential damping absorber.

In case PML field absorber is used together with absorbing particle boundaries, a special damping is applied for current density values in the PML area.
This treatment is to smooth effects of charges leaving the simulation volume and thus to better represent open boundaries.

For the exponential absorber, thickness of about 32 cells is recommended.
