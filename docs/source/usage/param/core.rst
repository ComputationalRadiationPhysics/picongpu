.. _usage-params-core:

PIC Core
--------

dimension.param
^^^^^^^^^^^^^^^

.. doxygenfile:: dimension.param
   :project: PIConGPU
   :path: include/picongpu/param/dimension.param
   :no-link:

grid.param
^^^^^^^^^^

.. doxygenfile:: grid.param
   :project: PIConGPU
   :path: include/picongpu/param/grid.param
   :no-link:

components.param
^^^^^^^^^^^^^^^^

.. doxygenfile:: components.param
   :project: PIConGPU
   :path: include/picongpu/param/components.param
   :no-link:

fieldSolver.param
^^^^^^^^^^^^^^^^^

.. doxygenfile:: fieldSolver.param
   :project: PIConGPU
   :path: include/picongpu/param/fieldSolver.param
   :no-link:

laser.param
^^^^^^^^^^^

.. doxygenfile:: laser.param
   :project: PIConGPU
   :path: include/picongpu/param/laser.param
   :no-link:

:ref:`List of available laser profiles <usage-params-core-laser-profiles>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   laser/profiles.rst

pusher.param
^^^^^^^^^^^^

.. doxygenfile:: pusher.param
   :project: PIConGPU
   :path: include/picongpu/param/pusher.param
   :no-link:

density.param
^^^^^^^^^^^^^

.. doxygenfile:: density.param
   :project: PIConGPU
   :path: include/picongpu/param/density.param
   :no-link:

speciesAttributes.param
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: speciesAttributes.param
   :project: PIConGPU
   :path: include/picongpu/param/speciesAttributes.param
   :no-link:

The following species attributes are defined by PMacc and always stored with a particle:

.. doxygenfile:: Identifier.hpp
   :project: PIConGPU
   :path: include/pmacc/particles/Identifier.hpp
   :no-link:

speciesConstants.param
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: speciesConstants.param
   :project: PIConGPU
   :path: include/picongpu/param/speciesConstants.param
   :no-link:

species.param
^^^^^^^^^^^^^

.. doxygenfile:: species.param
   :project: PIConGPU
   :path: include/picongpu/param/species.param
   :no-link:

speciesDefinition.param
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: speciesDefinition.param
   :project: PIConGPU
   :path: include/picongpu/param/speciesDefinition.param
   :no-link:

particle.param
^^^^^^^^^^^^^^

.. doxygenfile:: particle.param
   :project: PIConGPU
   :path: include/picongpu/param/particle.param
   :no-link:

More details on the order of initialization of particles inside a particle species :ref:`can be found here <usage-params-core-particles>`.

:ref:`List of all pre-defined particle manipulators <usage-params-core-particles-manipulation>`.

unit.param
^^^^^^^^^^

.. doxygenfile:: unit.param
   :project: PIConGPU
   :path: include/picongpu/simulation_defines/param/unit.param
   :no-link:

particleFilters.param
^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: particleFilters.param
   :project: PIConGPU
   :path: include/picongpu/simulation_defines/param/particleFilters.param
   :no-link:

:ref:`List of all pre-defined particle filters <usage-params-core-particles-filters>`.

speciesInitialization.param
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: speciesInitialization.param
   :project: PIConGPU
   :path: include/picongpu/param/speciesInitialization.param
   :no-link:

:ref:`List of all initialization methods for particle species <usage-params-core-particles-init>`.

.. toctree::
   :maxdepth: 1
   :hidden:

   particles/init.rst
