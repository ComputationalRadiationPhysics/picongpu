Configuration Header for *cupla*
=============================

cupla provides configuration header to use the library without CMake, e.g in a [cling](https://github.com/root-project/cling) environment.


Available Pre-compiler Definitions 
=================================

Pre-compiler definitions can be used to control the behavior of cupla/alpaka.
The definitions must be passed via a compiler flag or be defined before the accelerator header is included.
The default value will be used if the configuration header is included without defining any of the following options.

- `CUPLA_STREAM_ASYNC_ENABLED`: `0` use synchronous streams (default), `1` use asynchronous streams
- `CUPLA_HEADER_ONLY`: `1` *cupla* will be used as header-only library (default), otherwise you must compile all `.cpp` files in [`src/`](https://github.com/ComputationalRadiationPhysics/cupla/tree/master/src)


linker dependencies
===================

Depending of the used accelerator you must link the library `pthread` and/or activate OpenMP support.


Example
=======

To select an accelerator you must include the corresponding accelerator header from [`cupla/config/`](https://github.com/ComputationalRadiationPhysics/cupla/tree/master/include/cupla/config)


```C++

// use synchronous streams for the accelerator
#define CUPLA_STREAM_ASYNC_ENABLED 0
// use cupla as header-only library
#define CUPLA_HEADER_ONLY 1
#include <cupla/config/CpuSerial.hpp>


int main()
{
    int* res_ptr_d = nullptr;
    cudaMalloc( (void**)&res_ptr_d, sizeof( int ) );

    return 0;
}
```
