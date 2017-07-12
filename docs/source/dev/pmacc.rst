Important pmacc Classes
=======================

This is very, very small selection of classes of interest to get you started.

.. note::
   Please help adding more Doxygen doc strings to the classes described below.
   As an example, here is a listing of possible extensive docs that new developers find are missing:
   https://github.com/ComputationalRadiationPhysics/picongpu/issues/776

Environment
-----------

.. doxygenclass:: pmacc::Environment
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

DataConnector
-------------

.. doxygenclass:: pmacc::DataConnector
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

DataSpace
---------

.. doxygenclass:: pmacc::DataSpace
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Vector
------

.. doxygenclass:: pmacc::Vector
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

SuperCell
---------

.. doxygenclass:: pmacc::SuperCell
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

GridBuffer
----------

.. doxygenclass:: pmacc::GridBuffer
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

SimulationFieldHelper
---------------------

.. doxygenclass:: pmacc::SimulationFieldHelper
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ParticlesBase
-------------

.. doxygenclass:: pmacc::ParticlesBase
   :members:
   :protected-members:
   :undoc-members:

ParticleDescription
-------------------

.. doxygenclass:: pmacc::ParticleDescription
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ParticleBox
-----------

.. doxygenclass:: pmacc::ParticleBox
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Frame
-----

.. doxygenclass:: pmacc::Frame
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

IPlugin
-------

.. doxygenclass:: pmacc::IPlugin
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

PluginConnector
---------------

.. doxygenclass:: pmacc::PluginConnector
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

SimulationHelper
----------------

.. doxygenclass:: pmacc::SimulationHelper
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ForEach
-------

.. doxygenstruct:: pmacc::algorithms::forEach::ForEach
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Kernel Start
------------

.. doxygenstruct:: pmacc::exec::Kernel
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
See for example PIConGPU's :ref:`density.param <usage-params-core>` for usage.

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
