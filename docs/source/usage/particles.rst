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

Some of the particle manipulators further take the functors and filter (below) as arguments to manipulate attributes of particle species.

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

Manipulation Filters
--------------------

IsHandleValid
^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::filter::IsHandleValid
   :project: PIConGPU

RelativeGlobalDomainPosition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::filter::RelativeGlobalDomainPosition
   :project: PIConGPU

Manipulation Functors
---------------------

Some of the particle operations above can further take the following functors as arguments to manipulate attributes of particle species:

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

.. doxygenstruct:: picongpu::particles::manipulators::unary::Drift
   :project: PIConGPU

RandomPosition
^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::unary::RandomPosition
   :project: PIConGPU

Temperature
^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::unary::Temperature
   :project: PIConGPU

Assign
^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::binary::Assign
   :project: PIConGPU

DensityWeighting
^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::binary::DensityWeighting
   :project: PIConGPU

ProtonTimesWeighting
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::binary::ProtonTimesWeighting
   :project: PIConGPU
