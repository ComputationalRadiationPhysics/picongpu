.. _usage-particles:

Particles
=========

Initialization
--------------

The following operations can be applied in the picongpu::particles::InitPipeline inside speciesInitialization.param:

CreateDensity
^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::CreateDensity
   :project: PIConGPU

DeriveSpecies
^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::DeriveSpecies
   :project: PIConGPU

Manipulate
^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::Manipulate
   :project: PIConGPU

ManipulateDeriveSpecies
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::ManipulateDeriveSpecies
   :project: PIConGPU

FillAllGaps
^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::FillAllGaps
   :project: PIConGPU

Manipulation Functors
---------------------

Some of the particle operations above can take the following functors as arguments to manipulate attributes of particle species.
A particle filter (see following section) is used to only manipulated selected particles of a species with a functor.

Free
^^^^

.. doxygenstruct:: picongpu::particles::manipulators::generic::Free
   :project: PIConGPU

FreeRng
^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::generic::FreeRng
   :project: PIConGPU

CopyAttribute
^^^^^^^^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::unary::CopyAttribute
   :project: PIConGPU

Drift
^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::unary::Drift
   :project: PIConGPU

RandomPosition
^^^^^^^^^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::unary::RandomPosition
   :project: PIConGPU

Temperature
^^^^^^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::unary::Temperature
   :project: PIConGPU

Assign
^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::binary::Assign
   :project: PIConGPU

DensityWeighting
^^^^^^^^^^^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::binary::DensityWeighting
   :project: PIConGPU

ProtonTimesWeighting
^^^^^^^^^^^^^^^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::binary::ProtonTimesWeighting
   :project: PIConGPU

Manipulation Filters
--------------------

Most of the particle functors shall operate on all valid particles, where filter::All is the default assumption.
One can limit the domain or subset of particles with filters such as the ones below (or define new ones).

All
^^^

.. doxygenstruct:: picongpu::particles::filter::All
   :project: PIConGPU

RelativeGlobalDomainPosition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::filter::RelativeGlobalDomainPosition
   :project: PIConGPU

Free
^^^^

.. doxygenstruct:: picongpu::particles::filter::generic::Free
   :project: PIConGPU

FreeRng
^^^^^^^

.. doxygenstruct:: picongpu::particles::filter::generic::FreeRng
   :project: PIConGPU

FreeTotalCellOffset
^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::filter::generic::FreeTotalCellOffset
   :project: PIConGPU

Define a New Particle Filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   Not yet implemented.
