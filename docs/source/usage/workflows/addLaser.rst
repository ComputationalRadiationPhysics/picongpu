.. _usage-workflows-addLaser:

Adding Laser
------------

.. sectionauthor:: Sergei Bastrakov

There are several alternative ways of adding an incoming laser (or any source of electromagnetic field) to a PIConGPU simulation:

#. selecting a laser profile in :ref:`laser.param <usage-params-core>`
#. selecting non-none incident field profiles for respective boundaries in :ref:`incidentField.param <usage-params-core>`
#. using field or current background in :ref:`fieldBackground.param <usage-params-core>`

These ways operate independently of one another, each has its features and limitations.
Beware that the laser is fully accurate only for the standard Yee field solver.
For other field solver types, a user should evaluate the inaccuracies introduced.
Incident field, field- and current background should be fully accurate for all field solvers.

Incident field is applied using the total field/scattered field formulation described :ref:`here <model-TFSF>`.

The functioning of the laser (the first way) is covered in more detail in the following class:

.. doxygenclass:: picongpu::fields::laserProfiles::acc::BaseFunctor
   :project: PIConGPU
