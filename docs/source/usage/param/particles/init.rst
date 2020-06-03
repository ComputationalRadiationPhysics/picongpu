.. _usage-params-core-particles:

Particles
"""""""""

Particles are defined in modular steps.
First, species need to be generally defined in :ref:`speciesDefinition.param <usage-params-core>`.
Second, species are initialized with particles in :ref:`speciesInitialization.param <usage-params-core>`.

The following operations can be applied in the ``picongpu::particles::InitPipeline`` of the latter:

.. _usage-params-core-particles-init:

Initialization
''''''''''''''

CreateDensity
~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::CreateDensity
   :project: PIConGPU

Derive
~~~~~~

.. doxygenstruct:: picongpu::particles::Derive
   :project: PIConGPU

Manipulate
~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::Manipulate
   :project: PIConGPU

ManipulateDerive
~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::ManipulateDerive
   :project: PIConGPU

FillAllGaps
~~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::FillAllGaps
   :project: PIConGPU

.. _usage-params-core-particles-manipulation:

Manipulation Functors
'''''''''''''''''''''

Some of the particle operations above can take the following functors as arguments to manipulate attributes of particle species.
A particle filter (see following section) is used to only manipulated selected particles of a species with a functor.

Free
~~~~

.. doxygenstruct:: picongpu::particles::manipulators::generic::Free
   :project: PIConGPU

FreeRng
~~~~~~~

.. doxygenstruct:: picongpu::particles::manipulators::generic::FreeRng
   :project: PIConGPU

FreeTotalCellOffset
~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::manipulators::unary::FreeTotalCellOffset
   :project: PIConGPU

CopyAttribute
~~~~~~~~~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::unary::CopyAttribute
   :project: PIConGPU

Drift
~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::unary::Drift
   :project: PIConGPU

RandomPosition
~~~~~~~~~~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::unary::RandomPosition
   :project: PIConGPU

Temperature
~~~~~~~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::unary::Temperature
   :project: PIConGPU

Assign
~~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::binary::Assign
   :project: PIConGPU

DensityWeighting
~~~~~~~~~~~~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::binary::DensityWeighting
   :project: PIConGPU

ProtonTimesWeighting
~~~~~~~~~~~~~~~~~~~~

.. doxygentypedef:: picongpu::particles::manipulators::binary::ProtonTimesWeighting
   :project: PIConGPU

.. _usage-params-core-particles-filters:

Manipulation Filters
''''''''''''''''''''

Most of the particle functors shall operate on all valid particles, where ``filter::All`` is the default assumption.
One can limit the domain or subset of particles with filters such as the ones below (or define new ones).

All
~~~

.. doxygenstruct:: picongpu::particles::filter::All
   :project: PIConGPU

RelativeGlobalDomainPosition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::filter::RelativeGlobalDomainPosition
   :project: PIConGPU

Free
~~~~

.. doxygenstruct:: picongpu::particles::filter::generic::Free
   :project: PIConGPU

FreeRng
~~~~~~~

.. doxygenstruct:: picongpu::particles::filter::generic::FreeRng
   :project: PIConGPU

FreeTotalCellOffset
~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::particles::filter::generic::FreeTotalCellOffset
   :project: PIConGPU
