.. _usage-workflows-addLaser:

Adding Laser
------------

.. sectionauthor:: Sergei Bastrakov

There are several alternative ways of adding an incoming laser (or any source of electromagnetic field) to a PIConGPU simulation:

#. selecting incident field profiles for respective boundaries in :ref:`incidentField.param <usage-params-core>`
#. using field or current background in :ref:`fieldBackground.param <usage-params-core>`

These ways operate independently of one another, each has its features and limitations.
Incident field- and laser profiles currently match one another.

Incident Field
""""""""""""""

Incident field is an external field source producing a wave propagating inwards the simulation volume.
The source is applied at a generating surface that is a boundary of an axis-aligned box located inside the simulation area.
The implementation is based on the total field/scattered field formulation described in detail :ref:`here <model-TFSF>`.
A user defines positioning of this box in the total domain in :ref:`incidentField.param <usage-params-core>`.

With properly chosen sources, the generated waves only propagate inwards the volume bounded by the generation surface and there is no considerable leakage or noise.
The generating surface is otherwise transparent for all signals, such as reflections of lasers from a target.
A typical setup includes a field absorber located outside of the surface (directly or at a distance).

The surface must be offset inwards relative to each boundary by at least the field absorber thickness along the boundary so that the generating surface is located in the internal area.
An exception to this requirement is made for simulations using the moving window.
Then the surface positions along window movement direction can be located outside of the initially simulated volume.
In this case, parts of the surface located outside of the currently simulated volume are treated as if they had zero incident field and it is user's responsibility to apply a source matching such a case.

For each of the generation planes ``XMin, XMax, YMin, YMax, ZMin, ZMax`` (the latter two for 3d) a user sets an incident profile, or a typelist of such profiles, to be applied.
In case a typelist is used, the result is a sum of all profiles in the list.

In principle, the same sources should be applied at the whole generating surface, not just at planes where the lasers enter.
Then, the generated incident field will only exist in the internal volume, with application at the opposite side compensating and effectively removing it.
Note that this effect is not related to having a field absorber, but a property of the total field/scattered field formulation.
In practice there may be some noise due to numerical dispersion or imprecise source formulation.
In this case, a user may apply sources only at the "enter" parts of the generating surface but not on the opposite side (which will then be transparent), and employ a field absorber if needed.

The configuration is done through parameter structures, depending on the profile type.
Both profiles and parameter structures generally match their laser counterparts.
The differences between matching incident field- and laser profiles are:

#. positioning of incident field is controlled for the generation plane and not via an internal member ``::initPlaneY``
#. incident field profiles do not have an extra time delay equal to :math:`\mathrm{initPlaneY} * \Delta y / c` as lasers do (when needed, other parameters could be adjusted to accomodate for the delay)
#. default initial phase is chosen so that the laser starts smoothly at the generation plane (for laser it is always for plane :math:`y = 0`)
#. incident field uses generalized coordinate system and treats transversal axes and parameters generically (explained in comments of the profile parameters in question)

Note that the profile itself only controls properties of the laser, but not where it will be applied to.
It is a combination of profile and particular plane that together will produce an inward-going laser adhering to the profile.
For pre-set profiles a proper orientation of the wave will be provided by internal implementation.
With the ``Free`` profile, it is on a user to provide functors to calculate incident fields and ensure the orientation for the boundaries it is applied to (however, it does not have to work for all boundaries, only the ones in question).
Please refer to :ref:`the detailed description <model-TFSF>` for setting up ``Free`` profile, also for the case when only one of the external fields is known in explicit form.
For a laser profile with non zero field amplitudes on the transversal borders of the profile e.g. defined by the profile ``Free`` without a transversal envelop the trait ``MakePeriodicTransversalHuygensSurfaceContiguous`` must be specialized and returning true to handle field periodic boundaries correctly.

Incident field is compatible to all field solvers, however using field solvers other than Yee requires a larger offset of the generating surface from absorber depending on the stencil width along the boundary axis.
As a rule of thumb, this extra requirement is (order of FDTD solver / 2 - 1) cells.
Additionally, the current implementation requires the generation surface to be located sufficiently far away from local domain boundaries.
The same rule of a thumb can be used, with offsets being at least that many cells away from domain boundaries.
Validity of the provided offsets with respect to both conditions is checked at run time.
