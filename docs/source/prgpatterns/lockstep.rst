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

Functors passed into lockstep routines can have three different parameter signatures.

* No parameter, if the work is not requiring the linear index within a domain: ``[&](){ }``


* An unsigned 32bit integral parameter if the work depends on indices within the domain ``range [0,domain size)``: ``[&](uint32_t const linearIdx){}``


* ``lockstep::Idx`` as parameter. lockstep::Idx is holing the linear index within the domain and meta information to access a context variables: ``[&](pmacc::mappings::threads::lockstep::Idx const idx){}``


pmacc helpers
-------------

.. doxygenstruct:: pmacc::lockstep::Config
   :project: PIConGPU

.. doxygenstruct:: pmacc::lockstep::Idx
   :project: PIConGPU

.. doxygenclass:: pmacc::lockstep::Worker
   :project: PIConGPU

.. doxygenstruct:: pmacc::lockstep::Variable
   :project: PIConGPU

.. doxygenclass:: pmacc::lockstep::ForEach
   :project: PIConGPU

Common Patterns
---------------

Create a Context Variable
^^^^^^^^^^^^^^^^^^^^^^^^^

A context variable is used to transfer information from a subsequent lockstep to another.
You can use a context variable ``lockstep::Variable``, similar to a temporary local variable in a function.
A context variable must be defined outside of ``ForEach`` and should be accessed within the functor passed to ``ForEach`` only.

* ... and initialize with the index of the virtual worker

.. code-block:: cpp

    // assume one dimensional indexing of threads within a block
    uint32_t const workerIdx = cupla::threadIdx(acc).x;
    constexpr uint32_t frameSize = 256;
    constexpr uint32_t numWorkers = 42;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
    auto vIdx = forEachParticleSlotInFrame(
        [](lockstep::Idx const idx) -> int32_t
        {
            return idx;
        }
    );

    // is equal to

    // assume one dimensional indexing of threads within a block
    uint32_t const workerIdx = cupla::threadIdx(acc).x;
    constexpr uint32_t frameSize = 256;
    constexpr uint32_t numWorkers = 42;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
    // variable will be uninitialized
    auto vIdx = lockstep::makeVar<int32_t>(forEachParticleSlotInFrame);
    forEachParticleSlotInFrame(
        [&](lockstep::Idx const idx)
        {
            vIdx[idx] = idx;
        }
    );

* To default initialize a context variable you can pass the arguments directly during the creation.

.. code-block:: cpp

    // assume one dimensional indexing of threads within a block
    uint32_t const workerIdx = cupla::threadIdx(acc).x;
    constexpr uint32_t frameSize = 256;
    constexpr uint32_t numWorkers = 42;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
    auto var = lockstep::makeVar<int32_t>(forEachParticleSlotInFrame, 23);


* Data from a context variable can be accessed within independent lock steps.
  A virtual worker has only access to there own context variable data.

.. code-block:: cpp

    // assume one dimensional indexing of threads within a block
    uint32_t const workerIdx = cupla::threadIdx(acc).x;
    constexpr uint32_t frameSize = 256;
    constexpr uint32_t numWorkers = 42;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
    auto vIdx = forEachParticleSlotInFrame(
        [](lockstep::Idx const idx) -> int32_t
        {
            return idx;
        }
    );

    // store old linear index into oldVIdx
    auto oldVIdx = forEachExample(
        [&](lockstep::Idx const idx) -> int32_t
        {
            int32_t old = vIdx[idx];
            printf("virtual worker linear idx: %u == %u\n", vIdx[idx], idx);
            vIdx[idx] += 256;
            return old;
        }
    );

    forEachExample(
        [&](lockstep::Idx const idx)
        {
            printf("nothing changed: %u == %u - 256 == %u\n", oldVIdx[idx], vIdx[idx], idx);
        }
    );


Collective Loop
^^^^^^^^^^^^^^^

* each worker needs to pass a loop N times
* in this example, there are more dates than workers that process them

.. code-block:: bash

    // `frame` is a list which must be traversed collectively
    while( frame.isValid() )
    {
        // assume one dimensional indexing of threads within a block
        uint32_t const workerIdx = cupla::threadIdx(acc).x;
        constexpr uint32_t frameSize = 256;
        constexpr uint32_t numWorkers = 42;
        auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
        forEachParticleSlotInFrame(
           [&](lockstep::Idx const idx)
           {
               // independent work, idx can be used to access a context variable
           }
        forEachParticleSlotInFrame(
           [&](uint32_t const linearIdx)
           {
               // independent work based on the linear index only
           }
       );
    }


Non-Collective Loop
^^^^^^^^^^^^^^^^^^^

* each *virtual worker* increments a private variable

.. code-block:: cpp

    // assume one dimensional indexing of threads within a block
    uint32_t const workerIdx = cupla::threadIdx(acc).x;
    constexpr uint32_t frameSize = 256;
    constexpr uint32_t numWorkers = 42;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
    auto vWorkerIdx = lockstep::makeVar<int32_t>(forEachParticleSlotInFrame, 0);
    forEachParticleSlotInFrame(
        [&](auto const idx)
        {
            // assign the linear index to the virtual worker context variable
            vWorkerIdx[idx] = idx;
            for(int i = 0; i < 100; i++)
                vWorkerIdx[idx]++;
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

    // assume one dimensional indexing of threads within a block
    uint32_t const workerIdx = cupla::threadIdx(acc).x;
    auto onlyMaster = lockstep::makeMaster(workerIdx);

    // manipulate shared memory
    onlyMaster(
        [&]( )
        {
            finished = true;
        }
    );

    /* important: synchronize now, in case upcoming operations (with
     * other workers) access that manipulated shared memory section
     */
    worker.sync();
