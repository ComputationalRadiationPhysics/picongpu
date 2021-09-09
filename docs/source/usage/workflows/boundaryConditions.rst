.. _usage-workflows-boundaryConditions:

Boundary Conditions
-------------------

.. sectionauthor:: Sergei Bastrakov

Two kinds of boundary conditions are supported: periodic and absorbing.
They are set in a :ref:`.cfg file <usage-tbg>` with option ``--periodic <x> <y> <z>``.
Value 0 corresponds to absorbing boundaries along the axis (used by default), 1 corresponds to periodic.
The same boundary condition kind is applied for all particles species and fields, on both sides.

Particles
"""""""""

For particles, the boundaries always match the global simulation area border.
By default, boundary kinds match the value of ``--periodic``.
For species with a particle pusher, it can be overridden with option `<prefix>_boundary <x> <y> <z>`.
However, currently we do not support options that do not match the default value.

By default, the particle boundaries are applied at the global domain boundaries.
For absorbing boundaries it means that particles will exist in the field absorbing area.
This may be undesired for simulations with Perfectly Matched Layers (see below).
A user can change the boundary application area by setting option `<prefix>_boundaryOffset <x> <y> <z>`.
It sets an offset inwards from the global domain boundary.
Periodic boundaries only allow 0 offset, other kinds support non-negative offsets.

The internal treatment of particles crossing a boundary is controlled by the ``boundaryCondition`` flag in :ref:`speciesDefinition.param <usage-params-core>`.
However, this option is for expert users and generally should not be modified.
To set physical boundary conditions, use the command-line option described above.

Fields
""""""

Periodic boundary conditions for fields do not allow customization or variants.
The rest of the section concerns absorbing boundaries.

For the absorbing boundaries, there is a virtual field absorber layer inside the global simulation area near its boundaries.
Field values in the layer are treated differently from the rest, by a combination of a field solver and a field absorber.
It causes field dynamics inside the absorber layer to differ from that in a vacuum.
Otherwise, the affected cells are a normal part of a simulation in terms of indexing, particle handling, and output.
It is recommended to avoid, as possible, having particles in the field absorber layer.
Ideally, only particles leaving the simulation area are present there, on their way to be absorbed.
Note that particle absorption happens at the external surface of the field absorber layer, matching the global simulation area border.

The field absorber mechanism and user-controlled parameters depend on the field absorber kind enabled.
It is controlled by command-line option ``--fieldAbsorber``.
For all absorber kinds, the parameters are controlled by :ref:`fieldAbsorber.param <usage-params-core>`.

By default, the Perfectly Matched Layer (PML) absorber is used.
For this absorber, thickness of 8 to 12 cells is recommended.
Other absorber parameters can generally be used with default values.
PML generally provides much better absorber qualities than the exponential damping absorber.

For the exponential absorber, thickness of about 32 cells is recommended.
