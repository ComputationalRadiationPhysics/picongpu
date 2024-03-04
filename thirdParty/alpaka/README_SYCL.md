# Using the SYCL back-end

> :warning: **The SYCL back-end is currently in an experimental state. Bugs are to be expected.** Please report any issue you encounter.

## Supported Compilers

At the moment alpaka's SYCL back-end can only be used together with Intel oneAPI (latest stable version) and supports x86 CPUs, Intel GPUs and Intel FPGAs.

## CMake mode

`icpx` is identified as `IntelLLVM` by CMake. Compilers not based on `icpx` will be detected by CMake which will then print an error message.

### General CMake options

* `alpaka_ACC_SYCL_ENABLE`: set to `ON` to enable the SYCL back-end. Requires the activation of at least one oneAPI hardware target (see below).
* `oneDPL_DIR`: always required. Set to the CMake path of your oneDPL installation. Example: `/opt/intel/oneapi/dpl/2022.1.0/lib/cmake/oneDPL`.

### Building for x86 64-bit CPUs

The following CMake flags can be set for CPUs:

* `alpaka_SYCL_ONEAPI_CPU`: set to `ON` to enable compilation for CPUs. Relies on the Intel OpenCL CPU runtime and ahead-of-time compiler, which support Intel and AMD CPUs.
* `alpaka_SYCL_ONEAPI_CPU_ISA`: the Intel ISA to compile for. Look at the possible `--march` options listed in the output of `opencl-aot --help`. Default: `avx2`.

### Building for Intel FPGAs

Note: Intel FPGAs cannot be targeted together with other Intel hardware. This is because of different compiler trajectories as explained in the Intel oneAPI documentation.

* `alpaka_SYCL_ONEAPI_FPGA`: set to `ON` to enable compilation for Intel FPGAs.
* `alpaka_SYCL_ONEAPI_FPGA_MODE`: the Intel FPGA compilation mode. Valid values are `emulation`, `simulation` and `hardware` which correspond to the Intel high-level synthesis targets with the same name. Default: `emulation`.
* `alpaka_SYCL_ONEAPI_FPGA_BOARD`: the Intel FPGA board to compile for. Ignored in `emulation` mode but important for `simulation` and `hardware` modes. Valid values are `pac_a10` (Arria 10 GX), `pac_s10` (Stratix 10 SX / D5005) and `pac_s10_usm` (Stratix 10 SX / D5005 with restricted USM). Default: `pac_a10`.
* `alpaka_SYCL_ONEAPI_FPGA_BSP`: the Intel FPGA board support package (BSP). Valid values are `intel_a10gx_pac` (Arria 10 GX) and `intel_s10sx_pac` (Stratix 10 SX / D5005). It is also possible to supply the full path to the BSP here if `aoc` is unable to look this up by itself. Note that the BSP must be chosen according to the selected board (see previous bullet point).

### Building for Intel GPUs

* `alpaka_SYCL_ONEAPI_GPU`: set to `ON` to enable compilation for Intel GPUs.
* `alpaka_SYCL_ONEAPI_GPU_DEVICES`: semicolon-separated list of one or more Intel GPUs to compile for. The possible values for the devices are listed in the [UsersManual](https://intel.github.io/llvm-docs/UsersManual.html#generic-options) under the flag `-fsycl-targets`. Default: `intel_gpu_pvc`.
  Note: currently only one target at a time can be specified (limitation of the Intel Compiler)

## Standalone mode

### General

Using the SYCL back-end always requires the following flags:

* `-fsycl` (compiler and linker)
* `-fsycl-standard=2020` (compiler)

Device-side printing is possible with `printf`, it calls `sycl::ext::oneapi::experimental::printf` that emulates the standard one. This is an extension of the SYCL standard, still in an experimental state, therefore may not always work correctly.

### Building for x86 64-bit CPUs

1. `#include <alpaka/standalone/CpuSycl.hpp>` in your C++ code.
2. Add the following flags:
  * `-fsycl-targets=spir64_x86_64` (compiler and linker): to enable CPU compilation. Note: If you are using multiple SYCL hardware targets (like CPU and GPU) separate them by comma here.
  * `-Xsycl-target-backend=spir64_x86_64 "-march=<ISA>"` (linker): to choose the Intel ISA to compile for. Check the output of `opencl-aot --help` and look for the possible values of the `--march` flag.

### Building for Intel FPGAs

1. `#include <alpaka/standalone/FpgaSyclIntel.hpp>` in your C++ code.
2. Add the following flags:
  * `-fintelfpga` (compiler and linker): to enable FPGA compilation. Note: This flag is not compatible with the `-fsycl-targets` flag required for the other possible targets; Intel FPGAs thus cannot be used together with other hardware targets.
  * `-DALPAKA_FPGA_EMULATION` (compiler): to notify alpaka about compiling for the Intel FPGA `emulation` target. Required for `-Xsemulator` and forbidden for `-Xssimulation` and `-Xshardware`.
  * `-Xsemulator` (compiler and linker): to compile for Intel's `emulation` high-level synthesis target. Mutually exclusive with `-Xssimulation` and `-Xshardware`.
  * `-Xssimulation` (compiler and linker): to compile for Intel's `simulation` high-level synthesis target. Mutually exclusive with `-Xsemulator` and `-Xshardware`.
  * `-Xshardware` (compiler and linker): to compile for Intel's `hardware` high-level synthesis target. Mutually exclusive with `-Xsemulator` and `-Xssimulation`.
  * `-Xsboard=<BSP>:<BOARD>` (compiler and linker): to compile for a specific FPGA board. Required when either `-Xssimulation` or `-Xshardware` have been passed (no effect for `-Xsemulator`). Possible combinations for `<BSP>:<BOARD>` are `intel_a10gx_pac:pac_a10` (Arria 10 GX), `intel_s10sx_pac:pac_s10` (Stratix 10 SX / D5005) and `intel_s10sx_pac:pac_s10_usm` (Stratix 10 SX / D5005 with restricted USM).

### Building for Intel GPUs

1. `#include <alpaka/standalone/GpuSyclIntel.hpp>` in your C++ code.
2. Add the following flags:
  * `-fsycl-targets=intel_gpu_pvc` (compiler and linker): to enable GPU compilation. Note: If you are using multiple SYCL hardware targets (like CPU and GPU) separate them by comma here.

## Using the SYCL back-end

### Choosing the Accelerator

In contrast to the other back-ends the SYCL back-end comes with multiple different accelerators which should be chosen according to your requirements:

* `alpaka::AccCpuSycl` for targeting Intel and AMD CPUs. In contrast to the other CPU back-ends this will use Intel's OpenCL implementation for CPUs under the hood.
* `alpaka::AccFpgaSyclIntel` for targeting Intel FPGAs.
* `alpaka::AccGpuSyclIntel` for targeting Intel GPUs.

These can be used interchangeably (some restrictions apply - see below) with the non-experimental alpaka accelerators to compile an existing alpaka code for SYCL-capable hardware.

### Restrictions

* The Intel FPGA back-end cannot be used together with the Intel CPU / GPU back-ends. This is because of the different compilation trajectory required for FPGAs and is unlikely to be fixed anytime soon. See [here](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/programming-interface/fpga-flow/why-is-fpga-compilation-different.html) for an explanation.
* Similar to the CUDA and HIP back-ends the SYCL back-end only supports up to three kernel dimensions.
* Some Intel GPUs do not support the `double` type for device code. alpaka will not check this.
  You can enable software emulation for `double` precision types with
  ```bash
  export IGC_EnableDPEmulation=1
  export OverrideDefaultFP64Settings=1
  ```
  See [Intel's FAQ](https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-double-precision-emulation-fp64) for more information.
* The FPGA back-end does not support atomics. alpaka will not check this.
* Device global variables (corresponding to `__device__` and `__constant__` variables in CUDA) are not supported in the SYCL back-end yet.
* Shared memory works but on the GPU it is very slow.
* The latest Intel OpenCL CPU runtime does not work properly. Some tests (`atomicTest`, `blockSharedTest`, `blockSharedSharingTest` and `warpTest`) fail with a `PI_ERROR_OUT_OF_RESOURCES`. The only runtime version that seems to work is 2022.14.8.0.04 (can be downloaded [here](https://github.com/intel/llvm/releases/download/2022-WW33/oclcpuexp-2022.14.8.0.04_rel.tar.gz) apart from a bug with `all_of_group` / `any_of_group` that requires the warp size being equal to the block size as a workaround.

### Choosing the sub-group size (warp size)

Most SYCL targets support multiple sub-group sizes. There is a trait to specify at compile time the sub-group size to use for a kernel. For example, if `MyKernel` requires a sub-group size of 32, this can be declared specialising the `alpaka::trait::WarpSize`:
```cpp
struct MyKernel { ... };

template<typename TAcc>
struct alpaka::trait::WarpSize<MyKernel, TAcc>
    : std::integral_constant<std::uint32_t, 32>
{
};
```
This can be extended to kernels that support multiple sub-group sizes at compile time:
```cpp
template<std::uint32_t TWarpSize>
struct MyKernel { ... };

template<std::uint32_t TWarpSize, typename TAcc>
struct alpaka::trait::WarpSize<MyKernel<TWarpSize>, TAcc>
    : std::integral_constant<std::uint32_t, TWarpSize>
{
};
```

The default behaviour, when no sub-group size is specified, is to let the back-end compiler pick the preferred size.

Before launching a kernel with a compile-time sub-group size the user should query the sizes supported by the device, and choose accordingly. If the device does not support the requested size, the SYCL runtime will throw a synchronous exception.

During just-in-time (JIT) compilation this guarantees that a kernel is compiled only for the sizes supported by the device. During ahead-of-time (AOT) compilation this is not enough, because the device is not known at compile time. The SYCL specification mandates that the back-end compilers should not fail if a kernel uses unsupported features, like unsupported sub-group sizes. Unfortunately the Intel OpenCL CPU and GPU compilers currently fail with a hard error. To work around this limitation, use the preprocessor macros defined when compiling AOT for the new SYCL targets to enable the compilation only for the sub-group sizes supported by each device.

Note: while the CPU OpenCL back-end supports a sub-group size of 64, Intel's SYCL implementation currently does not. To avoid issues with the sub-group primitives, alpaka always considers the sub-group size of 64 as not supported by the device.
