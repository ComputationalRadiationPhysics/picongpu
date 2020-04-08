.. _usage-params-core-currentdeposition:

Current Deposition
""""""""""""""""""

The current solver can be set in :ref:`species.param <usage-params-core>` or directly per species :ref:`speciesDefinition.param <usage-params-core>`.

.. _usage-params-core-particles-currentsolver:

Current Solver
''''''''''''''

Esirkepov
~~~~~~~~~

.. doxygenstruct:: picongpu::currentSolver::Esirkepov
   :project: PIConGPU

EmZ
~~~

.. doxygenstruct:: picongpu::currentSolver::EmZ
   :project: PIConGPU

VillaBune
~~~~~~~~~

.. doxygenstruct:: picongpu::currentSolver::VillaBune
   :project: PIConGPU

EsirkepovNative
~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::currentSolver::EsirkepovNative
   :project: PIConGPU


.. _usage-params-core-particles-depositionstrategy:

Deposition Strategy
'''''''''''''''''''

A current solver supports a strategy to change how the algorithm behaves on different compute architectures.
The strategy is optional, could affect performance.

StridedCachedSupercells
~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::currentSolver::strategy::StridedCachedSupercells
   :project: PIConGPU

CachedSupercells
~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::currentSolver::strategy::CachedSupercells
   :project: PIConGPU

NonCachedSupercells
~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: picongpu::currentSolver::strategy::NonCachedSupercells
   :project: PIConGPU
