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
The treatment of particles crossing a boundary is controlled by the ``boundaryCondition`` flag in :ref:`speciesDefinition.param <usage-params-core>`.
Its default value provides the conventional treatment: particles crossing an absorbing boundary are deleted, particles crossing a periodic boundary are transferred to the other side of the global simulation area.
The behavior for both boundary kinds can be customized per species by changing the flag to a user-defined type.

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

The field absorber mechanism and user-controlled parameters depend on the field solver enabled.

Field solvers ``Yee<>``, ``Lehe<>``, and ``ArbitraryOrderFDTD<>`` use the exponential damping absorber.
Its parameters are controlled by :ref:`grid.param <usage-params-core>`.
Absorber thickness of about 32 cells is recommended.

Perfectly Matched Layer (PML) is used with field solvers ``YeePML<>``, ``LehePML<>``, and ``ArbitraryOrderFDTDPML<>``.
There is no other difference to their non-PML counterparts.
PML parameters are controlled by :ref:`pml.param <usage-params-core>`.
Absorber thickness of 8 to 12 cells is recommended, other parameters can generally be used with default values.
PML generally provides much better absorber qualities than the exponential damping absorber.
