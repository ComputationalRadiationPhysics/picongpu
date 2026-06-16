Usage
=====

Step 1: include
---------------

There is one header file that will include *all* necessary files:

```c++
#include <mallocMC/mallocMC.hpp>
```

Step 2a: choose policies
-----------------------

Each instance of a policy based allocator is composed through 5 **policies**.
Each policy is expressed as a **policy class**.

Currently, there are the following policy classes available:

|Policy                 | Policy Classes (implementations) | description |
|-------                |----------------------------------| ----------- |
|**CreationPolicy**     | Scatter`<conf1,conf2>`           | A scattered allocation to tradeoff fragmentation for allocation time, as proposed in [ScatterAlloc](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604). `conf1` configures the heap layout, `conf2` determines the hashing parameters|
|                       | FlatterScatter`<conf1,conf2>`    | Another scattered allocation algorithm similar in spirit to `Scatter` but with a flatter hierarchy and stronger concurrency invariants. `conf1` and `conf2` act as before.
|                       | OldMalloc                        | Device-side malloc/new and free/delete syscalls as implemented on the given device.
|**DistributionPolicy** | XMallocSIMD`<conf>`              | SIMD optimization for warp-wide allocation on NVIDIA CUDA accelerators, as proposed by [XMalloc](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5577907). `conf` is used to determine the pagesize. If used in combination with *Scatter*, the pagesizes must match |
|                       | Noop                             | no workload distribution at all |
|**OOMPolicy**          | ReturnNull                       | pointers will be *nullptr*, if the request could not be fulfilled |
|                       | ~~BadAllocException~~            | will throw a `std::bad_alloc` exception. The accelerator has to support exceptions |
|**ReservePoolPolicy**  | AlpakaBuf                        | Allocate a fixed-size buffer in an `alpaka`-provided container. |
|                       | CudaSetLimits                    | call to `CudaSetLimits` to increase the available Heap (e.g. when using *OldMalloc*) |
|**AlignmentPolicy**    | Shrink`<conf>`                   | shrinks the pool so that the starting pointer is well aligned, applies padding to requested memory chunks. `conf` is used to determine the alignment|
|                       | Noop                             | no alignment at all |

The user has to choose one of each policy that will form a useful allocator
(see [here](Usage.md#2c-combine-policies))

Step 2b: configure policies
---------------------------

Some of those policies are templates that can be configured through a
configuration struct. The default struct can be accessed through
```PolicyNamespace::PolicyClass<>::Properties```, which allows to
inherit a struct to modify some of its parameters before passing it
to the policy class:

```c++
// configure the AlignmentPolicy "Shrink"
struct ShrinkConfig : mallocMC::AlignmentPolicies::Shrink<>::Properties {
  static constexpr auto dataAlignment = 16;
};
```

Step 2c: combine policies
-------------------------

After configuring the chosen policies, they can be used as template
parameters to create the desired allocator type:

```c++
using namespace mallocMC;

using Allocator1 = mallocMC::Allocator<
  CreationPolicy::OldMalloc,
  DistributionPolicy::Noop,
  OOMPolicy::ReturnNull,
  ReservePoolPolicy::CudaSetLimits,
  AlignmentPolicy::Noop
>;
```

`Allocator1` will resemble the behaviour of classical device-side allocation known
from NVIDIA CUDA since compute capability sm_20. To get a more novel allocator, one
could create the following alias instead:

```c++
using namespace mallocMC;

using ScatterAllocator = mallocMC::Allocator<
  CreationPolicies::Scatter<>,
  DistributionPolicies::XMallocSIMD<>,
  OOMPolicies::ReturnNull,
  ReservePoolPolicies::SimpleCudaMalloc,
  AlignmentPolicies::Shrink<ShrinkConfig>
>;
```

Notice, how the policy classes `Scatter` and `XMallocSIMD` are instantiated without
template arguments to use the default configuration. `Shrink` however uses the
configuration struct defined above.

Step 3: instantiate allocator
-----------------------------

To use the defined allocator type, create an instance with the desired heap size:

```c++
ScatterAllocator sa( 512U * 1024U * 1024U ); // heap size of 512MiB
```

The allocator object offers the following methods

| Name | description |
|---------------------- |-------------------------|
| getAllocatorHandle()   | Acquire a handle from the allocator that can be used in kernels to allocate memory on device.
| getAvailableSlots(size_t)   | Determines number of allocatable slots of a certain size. This only works, if the chosen CreationPolicy supports it (can be found through `mallocMC::Traits<ScatterAllocator>::providesAvailableSlots`) |

One should note that on a running system with multiple threads manipulating
memory the information provided by `getAvailableSlots` is stale the moment it's
acquired and so relying on this information to be accurate is not recommended.
It is supposed to be used in initialisation/finalisation phases without dynamic
memory allocations or in tests.

Step 4: use dynamic memory allocation in a kernel
-------------------------------------------------

A handle to the allocator object must be passed to each kernel. The handle type is defined as an internal type of the allocator. Inside the kernel, this handle can be used to request memory.

The handle offers the following methods:

| Name | description |
|---------------------- |-------------------------|
| malloc(size_t) | Allocates memory on the accelerator  |
| free(size_t)     | Frees memory on the accelerator    |
| getAvailableSlots()   | Determines number of allocatable slots of a certain size. This only works, if the chosen CreationPolicy supports it (can be found through `mallocMC::Traits<ScatterAllocator>::providesAvailableSlots`).|

The comments on `getAvailableSlots` from above hold all the same.
A simplistic example would look like this:

```c++
#include <mallocMC/mallocMC.hpp>

namespace mallocMC = MC;

using ScatterAllocator = MC::Allocator<
  MC::CreationPolicies::Scatter<>,
  MC::DistributionPolicies::XMallocSIMD<>,
  MC::OOMPolicies::ReturnNull,
  MC::ReservePoolPolicies::SimpleCudaMalloc,
  MC::AlignmentPolicies::Shrink<ShrinkConfig>
>;

__global__ exampleKernel(ScatterAllocator::AllocatorHandle sah)
{
  // some code ...

  int* a = (int*) sah.malloc(sizeof(int)*42);

  // some more code, using *a

  sah.free(a);
}

int main(){
  ScatterAllocator sa( 1U * 512U * 1024U * 1024U ); // heap size of 512MiB
  exampleKernel<<< 32, 32 >>>(sa);

  return 0;
}
```

For more usage examples, have a look at the [examples](examples).
