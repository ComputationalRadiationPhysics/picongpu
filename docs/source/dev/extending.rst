.. _development-extending:

Extending PIConGPU
==================

.. sectionauthor:: Sergei Bastrakov

.. note::

   A number of places in :ref:`.param files <usage-params>` allow providing user-defined functors.
   The processing logic can largely be customized on that level, without modifying the source code.
   Such an external way is recommended when applicable.
   This section covers the case of extending the internal implementation.

General Simulation Loop Structure
---------------------------------

A PIConGPU simulation effectively boils down to performing initialization and then executing a simulation loop.
The simulation loop iterates over the requested number of time steps.
For each time step, first all enabled plugins are called.
Then all core computational stages are executed.
Details of creating a new plugin or a new stage are presented below.

Adding a Plugin
---------------

PIConGPU plugins can perform computations or output, and often do both.
Since all plugins are called at iteration start, they normally implement operations that are independent of the place in the computational loop.
For operations requiring a specific ordering, please refer to the next section.

To add a new plugin, make a new file or subdirectory inside ``include/picongpu/plugins``.
Each plugin class must inherit our base class ``ISimulationPlugin``.
In turn, it largely uses its own base class ``pmacc::ISimulationPlugin``.
These classes define the interface of all PIConGPU plugins.
The methods that most plugins want to override are:

* ``pluginRegisterHelp()`` adds command-line options for the plugin. In case a plugin introduces some new compile-time parameters, they are normally put to a new ``.param`` file.
* ``pluginGetName()`` sets a text name for the plugin, used to report errors currently. 
* ``pluginLoad()`` initializes internal data of the plugin after the command-line arguments are submitted. Note that a constructor of the plugin class would be called before that and so often cannot do a full initialization. Is called once upon simulation start.
* ``pluginUnload()`` finalizes internal data if necessary, is called once at the end of the simulation.
* ``setMappingDescription()`` is used to pass simulation data to be used in kernels
* ``notify()`` runs the plugin for the given time iteration. This method implements the computational logic of the plugin. It often involves calling an internal algorithm or writing an own kernel. Those are described in the following sections.
* ``checkpoint()`` saves plugin internal data for checkpointing if necessary
* ``restart()`` loads the internal data from a checkpoint (necessary if ``checkpoint()`` writes some data)

Most plugins are run with a certain period in time steps.
In this case, ``Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod)`` can be used inside ``pluginLoad()`` to set this period.
Then ``notify()`` will only be called for the active time steps.

For plugins (and most PIConGPU code) dealing with particles, it is common to template-parametrize based on species.
Such plugins should use base class ``plugins::multi::IInstance``.
There is also a helper class ``plugins::multi::IHelp`` for command-line parameters prefixed for species.
To match a plugin to applicable species, partially specialize trait ``particles::traits::SpeciesEligibleForSolver``.

Regardless of the base class used, the new plugin class must be instantiated at ``picongpu::PluginController`` and the new headers included there.
In case the plugin is conditionally available (e.g. requires an optional dependency), guards must also be placed there, around the include and instantiation.

When adding a plugin, do not forget to extend the documentation with plugin description and parameters.
At least, extend the list of :ref:`plugins <usage-plugins>` and :ref:`command-line parameters <usage-tbg>` (the latter via ``TBG_macros.cfg``).
Other welcome additions for a new plugin include a dedicated documentation, a new example demonstrating usage in a realistic scenario, and a Python postprocessing script.

Adding a Simulation Stage
-------------------------

The currently executed simulation stages and their order are defined inside ``Simulation::runOneStep()``.
All stage classes are in ``namespace picongpu::simulation::stage`` and their implementations are located in respective directory.
The existing stages share compatible interface and thus follow a pseudo-concept, not formally defined currently.
The interface is also compatible to functors passed to ``InitPipeline`` and ``IterationStartPipeline``.
Note that stage classes are just wrappers around the internal implementations.
Their purpose is to offer a high-level view of the steps and data dependencies in the computational loop, and add command-line parameters if necessary.

To add a new simulation stage:

* create a new ``.hpp`` file in ``picongpu/simulation/stage``.
* write your stage functor class following the interface of other stage functors.
* if needed (e.g. for command-line parameters or static data) store an instance of it as a member of ``Simulation`` class. 
* add a call to it inside ``Simulation::runOneStep()``.

As mentioned above, a simulation stage should merely be calling an actual implementation located at an appropriate place inside PIConGPU or pmacc.
When possible, it is recommended to use existing generic algorithms like ``particles::Manipulate<>`` or ``FieldTmp::computeValue()``.
Otherwise, one often has to implement an own kernel.

Writing a Kernel
----------------

Computational kernels are written using library `alpaka <https://github.com/alpaka-group/alpaka>`_.
Most kernel functors are templates parametrized with the number of threads per block, often called ``numWorkers``.
Kernel invocations are wrapped into a helper macro ``PMACC_KERNEL`` or ``PMACC_LOCKSTEP_KERNEL``.

A vast majority of PIConGPU kernels operate on two levels of parallelism: between supercells and inside each supercell.
This parallel pattern is covered by the mapper concept.

.. doxygenclass:: pmacc::MapperConcept
   :project: PIConGPU

For this parallel pattern, a mapper object provides the number of blocks to use for a kernel.
On the device side, the object provides a mapping between alpaka blocks and supercells to be processed.
Parallelism for threads between blocks is done inside the kernel.
It is often over cells in a supercell or particles in a frame using :ref:`lockstep programming <prgpatterns-lockstep>`.

A kernel often takes one or several data boxes from the host side.
The data boxes allow array-like access to data.
A more detailed description of boxes and other widely used classes is given in the following sections.
