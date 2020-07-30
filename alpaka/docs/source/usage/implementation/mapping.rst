Mapping onto Specific Hardware Architectures
============================================

By providing an accelerator independent interface for kernels, their execution and memory accesses at different hierarchy levels, *alpaka* allows the user to write accelerator independent code that does not neglect performance.

The mapping of the decomposition to the execution environment is handled by the back-ends provided by the *alpaka* library as well as user defined back-ends.
A computation that is described with a maximum of the parallelism available in the *redundant hierarchical parallelism* abstraction can not be mapped one to one to any existing hardware.
GPUs do not have vector registers for ``float`` or ``double`` types.
Therefore, the element level is often omitted on *CUDA* accelerators.
CPUs in turn are not (currently) capable of running thousands of threads concurrently and do not have equivalently fast inter-thread synchronization and shared memory access as GPUs do.

A major point of the *redundant hierarchical parallelism* abstraction is to ignore specific unsupported levels and utilize only the ones supported on a specific accelerator.
This allows a mapping to various current and future accelerators in a variety of ways enabling optimal usage of the underlying compute and memory capabilities.

The grid level is always mapped to the whole device being in consideration.
The scheduler can always execute multiple kernel grids from multiple queues in parallel by statically or dynamically subdividing the available resources.
However, this will only ever simplify the mapping due to less available processing units.
Furthermore, being restricted to less resources automatically improves the locality of data due to spatial and temporal locality properties of the caching hierarchy.

.. toctree::
   :maxdepth: 1

   mapping/CUDA
   mapping/x86
   mapping/accelerators
