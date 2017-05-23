.. _prgpatterns-lockstep:

.. seealso::

   In order to follow this section, you need to understand the `CUDA programming model <http://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model>`_.

Lockstep Programming Model
==========================

.. sectionauthor:: René Widera, Axel Huebl

The *lockstep programming model* structures code that is evaluated collectively and independently by workers (physical threads).
Actual processing is described by one-dimensional index domains of *virtual workers* which can even be changed within a kernel.
Mathematically, index domains are none-injective, total functions on physical workers.

An index domain is **independent** from data but **can** be mapped to a data domain, e.g. one to one or with more complex mappings.

Code which is implemented by the *lockstep programming model* is free of any dependencies between the number of worker and processed data elements.
To simplify the implementation, each index within a domain can be seen as a *virtual worker* which is processing one data element (like the common workflow to programming CUDA).
Each *worker* :math:`i` can be executed as :math:`N_i` *virtual workers* (:math:`1:N_i`).

PMacc helpers
-------------

.. doxygenstruct:: PMacc::mappings::threads::IdxConfig
   :project: PIConGPU

.. doxygenstruct:: PMacc::memory::CtxArray
   :project: PIConGPU

.. doxygenstruct:: PMacc::mappings::threads::ForEachIdx
   :project: PIConGPU

Common Patterns
---------------

Collective Loop
^^^^^^^^^^^^^^^

* each worker needs to pass a loop N times
* in this example, there are more dates than workers that process them

.. code-block:: bash

    // `frame` is a list which must be traversed collectively
    while( frame.isValid() )
    {
        uint32_t const workerIdx = threadIdx.x;
        using ParticleDomCfg = IdxConfig<
            frameSize,
            numWorker
        >;
        ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );
        forEachParticle(
           [&]( uint32_t const linearIdx, uint32_t const idx )
           {
               // independent work
           }
       );
    }


Non-Collective Loop
^^^^^^^^^^^^^^^^^^^

* each *virtual worker* increments a private variable

.. code-block:: cpp

    uint32_t const workerIdx = threadIdx.x;
    using ParticleDomCfg = IdxConfig<
        frameSize,
        numWorkers
    >;
    ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );
    memory::CtxArray< int, ParticleDomCfg > vWorkerIdx( 0 );
    forEachParticle(
        [&]( uint32_t const linearIdx, uint32_t const idx )
        {
            vWorkerIdx[ idx ] = linearIdx;
            for( int i = 0; i < 100; i++ )
                vWorkerIdx[ idx ]++;
        }
    );


Create a Context Variable
^^^^^^^^^^^^^^^^^^^^^^^^^

* ... and initialize with the index of the virtual worker

.. code-block:: cpp

    uint32_t const workerIdx = threadIdx.x;
    using ParticleDomCfg = IdxConfig<
        frameSize,
        numWorkers
    >;
    memory::CtxArray< int, ParticleDomCfg > vIdx(
        workerIdx,
        [&]( uint32_t const linearIdx, uint32_t const ) -> int32_t
        {
            return linearIdx;
        }
    );

    // is equal to

    memory::CtxArray< int, ParticleDomCfg > vIdx;
    ForEachIdx< ParticleDomCfg > forEachParticle{ workerIdx }(
        [&]( uint32_t const linearIdx, uint32_t const idx )
        {
            vIdx[ idx ] = linearIdx;
        }
    );


Using a Master Worker
^^^^^^^^^^^^^^^^^^^^^

* only one *virtual worker* (called *master*) of all available ``numWorkers`` manipulates a shared data structure for all others

.. code-block:: cpp

    // example: allocate shared memory (uninitialized)
    PMACC_SMEM(
        finished,
        bool
    );

    uint32_t const workerIdx = threadIdx.x;
    ForEachIdx<
        IdxConfig<
            1,
            numWorkers
        >
    > onlyMaster{ workerIdx };

    // manipulate shared memory
    onlyMaster(
        [&](
            uint32_t const,
            uint32_t const
        )
        {
            finished = true;
        }
    );

    /* important: synchronize now, in case upcoming operations (with
     * other workers) access that manipulated shared memory section
     */
    __syncthreads();
