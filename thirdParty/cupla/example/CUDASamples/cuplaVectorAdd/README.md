# vector add example with native cupla interface

This example is equal to `vectorAdd` but is not relying on the compatibility header included with (`cuda_to_cupla.hpp`) 
to allow the usage of CUDA function names and types.

CUDA prefixed functions/types are prefix with cupla instead.
CUDA functions/types those are not prefixed life in the namespace `cupla`.
Functions call need always the current used accelerator instance.
Non standard global variables like `threadIdx`, `blockDim` should be used as functions from the namespace `cupla`.