.. _usage-workflows-tracerParticles:

Tracer Particles
----------------

.. sectionauthor:: Axel Huebl

Tracer particles are like :ref:`probe particles <usage-workflows-probeParticles>`, but interact self-consistenly with the simulation.
They are usually used to visualize *representative* particle trajectories of a larger distribution.

In PIConGPU each species can be a tracer particle species, which allows tracking fields along the particle trajectory.

Workflow
""""""""

* ``speciesDefinition.param``:

Add the particle attribute ``particleId`` to your species to give each particle a unique id.
The id is optional and only required if the particle should be tracked over time.
Adding ``probeE`` creates a tracer species stores the interpolated electric field seen by the tracer particle.

  .. code-block:: cpp

      using ParticleFlagsTracerElectrons = MakeSeq_t<
          particlePusher< particles::pusher::Boris >,
          shape< UsedParticleShape >,
          interpolation< UsedField2Particle >,
          currentSolver::Esirkepov<UsedParticleCurrentSolver>
      >;

      using TracerElectron = Particles<
          PMACC_CSTRING( "e_tracer" ),
          ParticleFlagsTracerElectrons,
          MakeSeq_t<
              position< position_pic >,
              momentum,
              weighting,
              particleId,
              probeE
          >
      >;

and add it to ``VectorAllSpecies``:

.. code-block:: cpp

   using VectorAllSpecies = MakeSeq_t<
       TracerElectron,
       // ...
   >;

* create tracer particles by either

  * ``speciesInitialization.param``: initializing a low percentage of your initial density inside this species or
  * ``speciesInitialization.param``: assigning the target (electron) species of an ion's ionization routine to the tracer species or
  * ``speciesInitialization.param``: moving some particles of an already initialized species to the tracer species (upcoming)

* ``fileOutput.param``: make sure the the tracer particles are part of ``FileOutputParticles``

.. code-block:: cpp

   // either all via VectorAllSpecies or just select
   using FileOutputParticles = MakeSeq_t< TracerElectron >;

The electron tracer species is equivalent to normal electrons, the initial density distribution can be configured in :ref:`speciesInitialization.param <usage-params-core>`.

Known Limitations
"""""""""""""""""

* currently, only the electric field :math:`\vec E` and the magnetic field :math:`\vec B` can be recorded
* we currently do not support time averaging
