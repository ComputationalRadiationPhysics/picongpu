.. _prgpatterns-lockstep:

Lockstep Programming Model
==========================

.. sectionauthor:: René Widera

The *lockstep programming model* separates code that is evaluated by workers (threads) collectively and independently.
A problem is described by index domains. A index domain is **independent** from data but **can** be mapped to a data domain one to one but also with more complex mappings.

Code which is implemented by the *lockstep programming model* is free of any dependencies between the number of worker and processed data elements.
To simplify the implementation each index within a domain can be seen as a *virtual worker* which is processing one data element (like the common workflow to programming CUDA). Each *worker i* can be executed as N_i *virtual workers* (1:N_i).

PMacc helpers
-------------

.. doxygenstruct:: PMacc::mappings::threads::IdxConfig
   :project: PIConGPU

.. doxygenstruct:: PMacc::memory::CtxArray
   :project: PIConGPU

.. doxygenstruct:: PMacc::mappings::threads::ForEachIdx
   :project: PIConGPU

Common Pattern
--------------

Collective Loop
^^^^^^^^^^^^^^^

* each worker needs to pass a loop N times)
* in this example, there are more dates then workers that process them

.. code-block:: bash

    // `frame` is a list which must be traversed collectively
    while ( frame.isValid() )
    {
        const uint32_t workerIdx = threadIdx.x;
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

    const uint32_t workerIdx = threadIdx.x;
    using ParticleDomCfg = IdxConfig<
        frameSize,
        numWorker
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

    const uint32_t workerIdx = threadIdx.x;
    using ParticleDomCfg = IdxConfig<
        frameSize,
        numWorker
    >;
    memory::CtxArray< int, ParticleDomCfg > vIdx(
        workerIdx,
        [&]( uint32_t const linearIdx, uint32_t const ) -> DataSpace<dim>
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
