[:arrow_up: Up](../Library.md)

Interface Usage
===============

Accelerator Executable Functions
--------------------------------

Functions that should be executable on an accelerator have to be annotated with the execution domain (one of `ALPAKA_FN_HOST`, `ALPAKA_FN_ACC` and `ALPAKA_FN_HOST_ACC`).
They most probably also require access to the accelerator data and methods, such as indices and extents as well as functions to allocate shared memory and to synchronize all threads within a block. 
Therefore the accelerator has to be passed in as a constant template reference variable as can be seen in the following code snippet.

```C++
template<
	typename TAcc>
ALPAKA_FN_ACC auto doSomethingOnAccelerator(
	TAcc const & acc/*,
	...*/)
-> void
{
	//...
}
```


Kernel Definition
-----------------

There is no difference between the kernel entry point function and any other accelerator function in *alpaka* except that the kernel entry point is required to be a function object which has to have the accelerator as first argument.
This means, the kernel is an object that has implemented the `operator()` member function and can be called like any other function.
The following code snippet shows a basic example kernel function object.

```C++
struct MyKernel
{
	template<
		typename TAcc>    // Templated on the accelerator type.
	ALPAKA_FN_ACC       // Macro marking the function to be executable on all accelerators.
  auto operator()(    // The function / kernel to execute.
		TAcc & acc/*,     // The specific accelerator implementation.
		...*/) const      // Must be 'const'.
	-> void
	{
		//...
	}
                      // Class can have members but has to be std::is_trivially_copyable.
                      // Classes must not have pointers or references to host memory!
};
```

The kernel function object is shared across all threads in all blocks.
Due to the block execution order being undefined, there is no safe and consistent way of altering state that is stored inside of the function object.
Therefore, the `operator()` of the kernel function object has to be `const` and is not allowed to modify any of the object members.


Index and Work Division
-----------------------

The `alpaka::workdiv::getWorkDiv` and the `alpaka::idx::getIdx` functions both return a vector of the dimensionality the accelerator has been defined with.
They are parametrized by the origin of the calculation as well as the unit in which the values are calculated.
For example, `alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)` returns a vector with the extents of the grid in units of threads.


Memory Management
-----------------

The memory allocation function of the *alpaka* library (`alpaka::mem::buf::alloc<TElem>(device, extents)`) is uniform for all devices, even for the host device.
It does not return raw pointers but reference counted memory buffer objects that remove the necessity for manual freeing and the possibility of memory leaks.
Additionally the memory buffer objects know their extents, their pitches as well as the device they reside on.
This allows buffers that possibly reside on different devices with different pitches to be copied only by providing the buffer objects as well as the extents of the region to copy (`alpaka::mem::view::copy(bufDevA, bufDevB, copyExtents`).

Kernel Execution
----------------

The following source code listing shows the execution of a kernel by enqueuing the execution task into a stream.

```C++
// Define the dimensionality of the task.
using Dim = alpaka::dim::DimInt<1u>;
// Define the type of the sizes.
using Size = std::size_t;
// Define the accelerator to use.
using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
// Select the stream type.
using Stream = a::stream::StreamCpuAsync;

// Select a device to execute on.
auto devAcc(a::dev::DevManT<Acc>::getDevByIdx(0));
// Create a stream to enqueue the execution into.
Stream stream(devAcc);

// Create a 1-dimensional work division with 256 blocks a 16 threads.
auto const workDiv(alpaka::workdiv::WorkDivMembers<Dim, Size>(256u, 16u);
// Create an instance of the kernel function object.
MyKernel kernel;
// Create the execution task.
auto const exec(alpaka::exec::create<Acc>(workDiv, kernel/*, arguments ...*/);
// Enqueue the task into the stream.
alpaka::stream::enqueue(stream, exec);
```

The dimensionality of the task as well as the type for index and extent sizes have to be defined explicitly.
Following this, the type of accelerator to execute on, as well as the type of the stream have to be defined.
For both of these types instances have to be created.
For the accelerator this has to be done indirectly by enumerating the required device via the device manager, whereas the stream can be created directly.

To execute the kernel, an instance of the kernel function object has to be constructed.
Following this, an execution task combining the work division (grid and block sizes) with the kernel function object and the bound invocation arguments has to be created.
After that this task can be enqueued into a stream for immediate or later execution (depending on the stream used).
