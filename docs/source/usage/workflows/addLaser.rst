.. _usage-workflows-addLaser:

Adding Laser
------------

.. sectionauthor:: Sergei Bastrakov

There are several alternative ways of adding an incoming laser (or any source of electromagnetic field) to a PIConGPU simulation:

#. selecting incident field profiles for respective boundaries in :ref:`incidentField.param <usage-params-core>`
#. selecting a laser profile in :ref:`laser.param <usage-params-core>`
#. using field or current background in :ref:`fieldBackground.param <usage-params-core>`

These ways operate independently of one another, each has its features and limitations.
Incident field- and laser profiles currently match one another.

Incident Field
""""""""""""""

Incident field is an external field source producing a wave propagating inwards the simulation volume.
The source is applied at a boundary of an axis-aligned box located inside the simulation area.
The implementation is based on the total field/scattered field formulation described in detail :ref:`here <model-TFSF>`.
A user sets offsets of each side of this box from global domain boundary in :ref:`incidentField.param <usage-params-core>`.
Each offset must cover at least the field absorber thickness along the boundary so that the generating surface is located in the internal area.

For each of the generation planes ``XMin, XMax, YMin, YMax, ZMin, ZMax`` (the latter two for 3d) a user sets incident profile to be applied.
The configuration is done through parameter structures, depending on the profile type.
Both profiles and parameter structures generally match their laser counterparts.
The differences between matching incident field- and laser profiles are:

# positioning of incident field is controlled for the generation plane and not via an internal member ``::initPlaneY``
# incident field profiles do not have an extra time delay equal to :math:`initPlaneY * \Delta y / c` as lasers do (when needed, other parameters could be adjusted to accomodate for the delay)
# default initial phase is chosen so that the laser starts smoothly at the generation plane (for laser it is always for plane :math:`y = 0`) 
# incident field uses generalized coordinate system and treats transversal axes and parameters generically (explained in comments of the profile parameters in question)

Note that the profile itself only controls properties of the laser, but not where it will be applied to.
It is a combination of profile and particular plane that together will produce an inward-going laser adhering to the profile.
For pre-set profiles a proper orientation of the wave will be provided by internal implementation.
With the ``Free`` profile, it is on a user to provide functors to calculate incident fields and ensure the orientation for the boundaries it is applied to (however, it does not have to work for all boundaries, only the ones in question).
Please refer to :ref:`the detailed description <model-TFSF>` for setting up ``Free`` profile, also for the case when only one of the external fields is known in explicit form.

Incident field is compatible to all field solvers, however using field solvers other than Yee requires a larger offset depending on the stencil width along the boundary axis.
As a rule of thumb, this extra requirement is (order of FDTD solver / 2 - 1).
Additionally, the current implementation requires the offset be located sufficiently far away from local domain boundaries.
The same rule of a thumb can be used, with offsets being at least that many cells away from domain boundaries.
Validity of the provided offsets with respect to both conditions is checked at run time.

Laser
"""""

Laser profiles are still supported, but deprecated.
Consider switching to the incident field counterpart of your laser profile as ``YMin`` source.
The transition should be straightforward, please refer to the previous section for differences from incident field.

Beware that the laser is fully accurate only for the standard Yee field solver.
For other field solver types, a user should evaluate the inaccuracies introduced.

The functioning of the laser (the second way) is covered in more detail in the following class:

.. doxygenclass:: picongpu::fields::laserProfiles::acc::BaseFunctor
   :project: PIConGPU
