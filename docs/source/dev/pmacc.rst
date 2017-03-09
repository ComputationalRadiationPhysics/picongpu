Important PMacc Classes
=======================

This is very, very small selection of classes of interest to get you started.

.. note::
   Please help adding more Doxygen doc strings to the classes described below.
   As an example, here is a listing of possible extensive docs that new developers find are missing:
   https://github.com/ComputationalRadiationPhysics/picongpu/issues/776

Environment
-----------

.. doxygenclass:: PMacc::Environment
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

DataConnector
-------------

.. doxygenclass:: PMacc::DataConnector
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

DataSpace
---------

.. doxygenclass:: PMacc::DataSpace
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Vector
------

.. doxygenclass:: PMacc::Vector
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

SuperCell
---------

.. doxygenclass:: PMacc::SuperCell
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

GridBuffer
----------

.. doxygenclass:: PMacc::GridBuffer
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

SimulationFieldHelper
---------------------

.. doxygenclass:: PMacc::SimulationFieldHelper
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ParticlesBase
-------------

.. doxygenclass:: PMacc::ParticlesBase
   :members:
   :protected-members:
   :undoc-members:

ParticleDescription
-------------------

.. doxygenclass:: PMacc::ParticleDescription
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ParticleBox
-----------

.. doxygenclass:: PMacc::ParticleBox
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Frame
-----

.. doxygenclass:: PMacc::Frame
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

IPlugin
-------

.. doxygenclass:: PMacc::IPlugin
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

PluginConnector
---------------

.. doxygenclass:: PMacc::PluginConnector
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

SimulationHelper
----------------

.. doxygenclass:: PMacc::SimulationHelper
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ForEach
-------

.. doxygenstruct:: PMacc::algorithms::forEach::ForEach
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Kernel Start
------------

.. doxygenstruct:: PMacc::exec::Kernel
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

.. doxygendefine:: PMACC_KERNEL
   :project: PIConGPU

Struct Factory
--------------

Syntax to generate structs with all members inline.
Allows to conveniently switch between variable and constant defined members without the need to declare or initialize them externally.
See for example PIConGPU's densityConfig.param for usage.

.. doxygendefine:: PMACC_STRUCT
   :project: PIConGPU

.. doxygendefine:: PMACC_C_VECTOR_DIM
   :project: PIConGPU

.. doxygendefine:: PMACC_C_VALUE
   :project: PIConGPU

.. doxygendefine:: PMACC_VALUE
   :project: PIConGPU

.. doxygendefine:: PMACC_VECTOR
   :project: PIConGPU

.. doxygendefine:: PMACC_VECTOR_DIM
   :project: PIConGPU

.. doxygendefine:: PMACC_C_STRING
   :project: PIConGPU

.. doxygendefine:: PMACC_EXTENT
   :project: PIConGPU

Identifier
----------

Construct unique types, e.g. to name, access and assign default values to particle species' attributes.
See for example PIConGPU's speciesAttributes.param for usage.

.. doxygendefine:: value_identifier
   :project: PIConGPU

.. doxygendefine:: alias
   :project: PIConGPU
