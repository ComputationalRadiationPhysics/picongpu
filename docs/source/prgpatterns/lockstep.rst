.. _prgpatterns-lockstep:

.. seealso::

   In order to follow this section, you need to understand the `CUDA programming model <http://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model>`_.

Lockstep Programming Model
==========================

.. sectionauthor:: René Widera, Axel Huebl

The *lockstep programming model* structures code that is evaluated collectively and independently by workers (physical threads) within a alpaka block.
Actual processing is described by one-dimensional index domains which are known compile time and can even be changed within a kernel.

An index domain is **independent** of the data but **can** be mapped to a data domain, e.g. one to one or with more complex mappings.
A index domain is processed collectively by all workers.

Code which is implemented by the *lockstep programming model* is free of any dependencies between the number of worker and processed data elements.
To simplify the implementation, each index within a domain can be mapped to a single data element (like the common workflow to programming CUDA).
But even within this simplified picture one real worker (i.e. physical thread) could still be assigned the workload of any number of domain indices.

Functors passed into lockstep routines can have three different base parameter signatures.
Additionally each case can be extended by an arbitrary number parameters to get access to context variables.

* No parameter, if the work is not requiring the linear index within a domain: ``[&](){ }``


* An unsigned 32bit integral parameter if the work depends on indices within the domain ``range [0,domain size)``: ``[&](uint32_t const linearIdx){}``


* ``lockstep::Idx`` as parameter. lockstep::Idx is holding the linear index within the domain and meta information to access a context variables: ``[&](pmacc::mappings::threads::lockstep::Idx const idx){}``

Context variables, over worker distributed arrays, can be passed as additional arguments to the lockstep foreach.
The corresponding data for each index element of the domain will be passed as additional argument to the lambda function.

The naming used for methods or members
--------------------------------------

* ``*DomSize`` is the index domain size as scalar value, typically an integral type
* ``*DomSizeND`` is the N-dimensional index domain size, typically of the type ``pmacc::math::Vector<>`` or ``pmacc::math::CT:Vector``
* ``*DomIdx``  is the index domain element as scalar value, typically an integral type
* ``*DomIdxND`` is the N-dimensional index domain element, typically of the type ``pmacc::math::Vector<>``
* ``*Size`` is the size of data as scalar value
* ``*SizeND`` is the N-dimensional data size, typically of the type ``pmacc::math::Vector<>``

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

* ... and initialize with the index of the domain element

.. code-block:: cpp

    // variable 'worker' is provided by pmacc if the kernel launch macro `PMACC_LOCKSTEP_KERNEL()` is used.
    constexpr uint32_t frameSize = 256;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
    auto elemIdx = forEachParticleSlotInFrame(
        [](lockstep::Idx const idx) -> int32_t
        {
            return idx;
        }
    );

    // is equal to

    // assume one dimensional indexing of threads within a block
    constexpr uint32_t frameSize = 256;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
    // variable will be uninitialized
    auto elemIdx = lockstep::makeVar<int32_t>(forEachParticleSlotInFrame);
    forEachParticleSlotInFrame(
        [&](uint32_t const idx, auto& vIndex)
        {
            vIndex = idx;
        },
        elemIdx
    );
    // is equal to
    forEachParticleSlotInFrame(
        [&](lockstep::Idx const idx)
        {
            elemIdx[idx] = idx;
        }
    );

* To default initialize a context variable you can pass the arguments directly during the creation.

.. code-block:: cpp

    // variable 'worker' is provided by pmacc if the kernel launch macro `PMACC_LOCKSTEP_KERNEL()` is used.
    constexpr uint32_t frameSize = 256;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
    auto var = lockstep::makeVar<int32_t>(forEachParticleSlotInFrame, 23);


* Data from a context variable can be accessed within independent lock steps.
  Only data elements those correspond to the element index of the domain can be accessed.

.. code-block:: cpp

    // variable 'worker' is provided by pmacc if the kernel launch macro `PMACC_LOCKSTEP_KERNEL()` is used.
    constexpr uint32_t frameSize = 256;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
    auto elemIdx = forEachParticleSlotInFrame(
        [](uint32_t const idx) -> int32_t
        {
            return idx;
        }
    );

    // store old linear index into oldElemIdx
    auto oldElemIdx = forEachExample(
        [&](lockstep::Idx const idx) -> int32_t
        {
            int32_t old = elemIdx[idx];
            printf("domain element idx: %u == %u\n", elemIdx[idx], idx);
            elemIdx[idx] += 256;
            return old;
        }
    );

    // To avoid convusion between read-only and read-write input variables we suggest using
    // const for read only variables.
    forEachExample(
        [&](lockstep::Idx const idx, int32_t const oldIndex, int32_t const vIndex)
        {
            printf("nothing changed: %u == %u - 256 == %u\n", oldIndex, vIndex, idx);
        },
        oldElemIdx,
        elemIdx
    );


Collective Loop over particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* each worker needs to pass a loop N times
* in this example, there are more dates than workers that process them

.. code-block:: bash

    // variable 'worker' is provided by pmacc if the kernel launch macro `PMACC_LOCKSTEP_KERNEL()` is used.
    // `frame` is a list which must be traversed collectively
    while( frame.isValid() )
    {
        // assume one dimensional indexing of threads within a block
        constexpr uint32_t frameSize = 256;
        auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
        forEachParticleSlotInFrame(
           [&](lockstep::Idx const idx)
           {
               // independent work, idx can be used to access a context variable
           }
        forEachParticleSlotInFrame(
           [&](uint32_t const linearIdx)
           {
               // independent work based on the linear index only, e.g. shared memory access
           }
       );
    }


Non-Collective Loop over particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* each element index of the domain increments a private variable

.. code-block:: cpp

    // variable 'worker' is provided by pmacc if the kernel launch macro `PMACC_LOCKSTEP_KERNEL()` is used.
    constexpr uint32_t frameSize = 256;
    auto forEachParticleSlotInFrame = lockstep::makeForEach<frameSize>(worker);
    auto vWorkerIdx = lockstep::makeVar<int32_t>(forEachParticleSlotInFrame, 0);
    forEachParticleSlotInFrame(
        [&](auto const idx, int32_t& vWorker)
        {
            // assign the linear element index to context variable
            vWorker = idx;
            for(int i = 0; i < 100; i++)
                vWorker++;
        },
        vWorkerIdx
    );


Using a Master Worker
^^^^^^^^^^^^^^^^^^^^^

* only a single element index of the domain (called *master*) manipulates a shared data structure for all others

.. code-block:: cpp

    // example: allocate shared memory (uninitialized)
    PMACC_SMEM(
        finished,
        bool
    );

    // variable 'worker' is provided by pmacc if the kernel launch macro `PMACC_LOCKSTEP_KERNEL()` is used.
    auto onlyMaster = lockstep::makeMaster(worker);

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

Practical Examples
------------------

If possible kernels should be written without assuming any lockstep domain size and number of alpaka blocks selected at the kernel start.
This ensure that the kernel results are always correct even if the user doesn't chose the right parameters for the kernel execution.

  .. literalinclude:: ../../../include/pmacc/test/lockstep/lockstepUT.cpp
     :language: C++
     :start-after: doc-include-start: lockstep generic kernel
     :end-before: doc-include-end: lockstep generic kernel
     :dedent:

The block domain size can also be derived from a instance of any object if the trait ``pmacc::lockstep::traits::MakeBlockCfg`` is defined.

  .. literalinclude:: ../../../include/pmacc/test/lockstep/lockstepUT.cpp
     :language: C++
     :start-after: doc-include-start: lockstep generic kernel buffer selected domain size
     :end-before: doc-include-end: lockstep generic kernel buffer selected domain size
     :dedent:

Sometimes it is not possible to write a generic kernel and a hard coded block domain size is required to fulfill stencil condition or other requirements.
In this case it is possible to use on device ``pmacc::lockstep::makeForEach<hardCodedBlockDomSize>(worker)``.
The problem is that the user needs to know this hard coded requirement during the kernel call else it could be the kernel is running slow.
It is possible that too many worker threads are idling during the execution because the selected block domain during the kernel call is larger than the required block domain within the kernel.
By defining the member variable ``blockDomSize`` and not providing the block domain size during the kernel configuration the kernel will
be executed automatically with the block domain size specialized by the kernel.
Overwriting the block domain size during the kernel execution is triggering a static assertion during compiling.

  .. literalinclude:: ../../../include/pmacc/test/lockstep/lockstepUT.cpp
     :language: C++
     :start-after: doc-include-start: lockstep generic kernel hard coded domain size
     :end-before: doc-include-end: lockstep generic kernel hard coded domain size
     :dedent:

Equally to the scalar block domain size ``blockDomSize`` a member type ``BlockDomSizeND`` of the pmacc type ``pmacc::math::CT::Uint32<>`` can be defined to express a N-dimensional block domain.
``blockDomSize`` and ``BlockDomSizeND`` are mutual exclusive and can not be defined at the same time for a kernel.

  .. literalinclude:: ../../../include/pmacc/test/lockstep/lockstepUT.cpp
     :language: C++
     :start-after: doc-include-start: lockstep generic kernel hard coded N dimensional domain size
     :end-before: doc-include-end: lockstep generic kernel hard coded N dimensional domain size
     :dedent:

To use dynamic shared memory within a lockstep kernel the kernel must be configured with ``configSMem`` instead of `config`

  .. literalinclude:: ../../../include/pmacc/test/lockstep/lockstepUT.cpp
     :language: C++
     :start-after: doc-include-start: lockstep generic kernel with dynamic shared memory
     :end-before: doc-include-end: lockstep generic kernel with dynamic shared memory
     :dedent:
