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

.. doxygenstruct:: pmacc::math::Vector
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

.. doxygenstruct:: pmacc::ParticleDescription
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

ParticleBox
-----------

.. doxygenclass:: pmacc::ParticlesBox
   :project: PIConGPU
   :members:
   :protected-members:
   :undoc-members:

Frame
-----

.. doxygenstruct:: pmacc::Frame
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

.. doxygenstruct:: pmacc::meta::ForEach
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

Identifier
----------

Construct unique types, e.g. to name, access and assign default values to particle species' attributes.
See for example PIConGPU's speciesAttributes.param for usage.

.. doxygendefine:: value_identifier
   :project: PIConGPU

.. doxygendefine:: alias
   :project: PIConGPU
