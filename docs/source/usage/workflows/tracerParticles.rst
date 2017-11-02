.. _usage-workflows-tracerParticles:

Tracer Particles
----------------

.. sectionauthor:: Axel Huebl

Tracer particles are like :ref:`probe particles <usage-workflows-probeParticles>`, but interact self-consistenly with the simulation.
They are usually used to visualize *representative* particle trajectories of a larger distribution.

Workflow
""""""""

* ``speciesDefinition.param``: create a species specifically for tracer particles

  * add the particle attribute ``particleId`` to your species' ``Particles< ... >`` class (third argument, ``T_Attributes``)
  * optional: add ``fieldE`` and ``fieldB`` attributes to the species to store fields as in :ref:`probes <usage-workflows-probeParticles>` 

* create tracer particles by either

  * ``speciesInitialization.param``: initializing a low percentage of your initial density inside this species or
  * ``speciesInitialization.param``: assigning the target (electron) species of an ion's ionization routine to the tracer species or
  * ``speciesInitialization.param``: moving some particles of an already initialized species to the tracer species (upcoming)

* ``fileOutput.param``: output the tracer particles

Known Limitations
"""""""""""""""""

* currently, only the electric field :math:`\vec E` and the magnetic field :math:`\vec B` can be recorded
* we currently do not support time averaging
