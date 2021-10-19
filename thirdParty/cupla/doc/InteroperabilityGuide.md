Use *cupla* together with other alpaka Based Software
=====================================================

Sometimes it is necessary to use an `alpaka` function for which there is no equivalent in `cupla`, or you want to use a library with `cupla` that was developed with an `alpaka` compatible interface, such as [vikunja](https://github.com/alpaka-group/vikunja/). This requires some `alpaka` types that are handled automatically and internally by `cupla`. The tutorial shows how to access these types.

Accessing Acc Mapping, Acc Type, Queue Type and Stream
======================================================

The first codes sample shows how `alpaka` generates the required accelerator types. The second code sample shows how to get the accelerator types in `cupla`.

```C++
#include <alpaka/alpaka.hpp>

using Dim = alpaka::DimInt<3>;
using Idx = std::size_t;

using Host = alpaka::AccCpuSerial<Dim, Idx>;
using Dev = alpaka::AccGpuCudaRt<Dim, Idx>;

using QueueProperty = alpaka::Blocking;

using HostQueue = alpaka::Queue<Host, QueueProperty>;
using DevQueue = alpaka::Queue<Dev, QueueProperty>;

auto const hostAcc = alpaka::getDevByIdx<Host>(0u);
HostQueue hostQueue(hostAcc);

auto const devAcc = alpaka::getDevByIdx<Dev>(0u);
DevQueue devQueue(devAcc);

// ...

alpaka::exec<Dev>(
	devQueue,
	workDiv,
	kernel);


```

```C++
#include <cupla.hpp>
#include <alpaka/alpaka.hpp>

cupla::AccHost const hostAcc(cupla::manager::Device<cupla::AccHost>::get().current());
cupla::AccHostStream hostQueue(cupla::manager::Stream<cupla::AccHost, cupla::AccHostStream>::get().stream(0));

cupla::AccDev const devAcc(cupla::manager::Device<cupla::AccDev>::get().current());
cupla::AccStream devQueue(cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(0));

// ...

alpaka::exec<cupla::Acc>(
	devQueue,
	workDiv,
	kernel);

```

Example
=======

An example of how `cupla` works with `vikunja`, which provides an alpaka compatible interface,
can be found here: [cupla-vikunja-reduce](https://github.com/alpaka-group/vikunja/tree/master/example/cuplaReduce)
