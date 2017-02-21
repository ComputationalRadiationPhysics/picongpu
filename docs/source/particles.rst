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


Manipulation
------------

Some of the particle operations above can further take the following functors as arguments to manipulate attributes of particle species:

AssignImpl
^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::AssignImpl
   :project: PIConGPU

CopyAttribute
^^^^^^^^^^^^^

.. doxygentypedef:: picongpu::particles::manipulators::CopyAttribute
   :project: PIConGPU

DensityWeighting
^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::DensityWeighting
   :project: PIConGPU

DriftImpl
^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::DriftImpl
   :project: PIConGPU

FreeImpl
^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::FreeImpl
   :project: PIConGPU

FreeRngImpl
^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::FreeRngImpl
   :project: PIConGPU

IfRelativeGlobalPositionImpl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::IfRelativeGlobalPositionImpl
   :project: PIConGPU

ProtonTimesWeighting
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::ProtonTimesWeighting
   :project: PIConGPU

RandomPositionImpl
^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::RandomPositionImpl
   :project: PIConGPU

SetAttributeImpl
^^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::SetAttributeImpl
   :project: PIConGPU

TemperatureImpl
^^^^^^^^^^^^^^^

.. doxygenstruct:: picongpu::particles::manipulators::TemperatureImpl
   :project: PIConGPU
