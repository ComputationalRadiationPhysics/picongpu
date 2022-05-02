# Using the SYCL back-end

> :warning: **The SYCL back-end is currently in an experimental state. Bugs are to be expected.** Please report any issue you encounter.

## Supported Compilers

alpaka's SYCL back-end can be used with the following compilers:

* Intel oneAPI DPC++ v2022.0 (supports Intel CPUs, GPUs, FPGAs)
* AMD/Xilinx' fork of the above (supports AMD/Xilinx FPGAs) which can be found [here](https://github.com/triSYCL/sycl).
  Please use the `sycl/unified/next` branch with its latest commit and be sure to follow the environment setup instructions found there.

Other versions of these compilers may work but are untested. All other SYCL compilers are untested and unlikely to work. All compilers have only been tested on Linux; Windows remains untested.

From this point forward it is assumed that you have installed one of the compilers listed above as well as any hardware-specific
dependencies for your desired platform.

## CMake mode

Because all supported compilers are identified as `Clang` by CMake 3.18 it is the responsibility of the user to provide a suitable SYCL compiler. It is recommended to do this by setting the `CXX` environment variable:

```bash
> CXX=/path/to/sycl/clang++ cmake <other options>
```

### General CMake options

* `alpaka_ACC_SYCL_ENABLE`: set to `ON` to enable the SYCL back-end. Requires the activation of at least one SYCL platform (see below).
* `alpaka_SYCL_ENABLE_IOSTREAM`: set to `ON` to enable device-side printing. Force-enabled if `BUILD_TESTING` is enabled.
* `alpaka_SYCL_IOSTREAM_KIB`: Kibibytes per block reserved as output buffer for device-side printing. This cannot exceed the amount of shared memory per block. Only takes effect if `alpaka_SYCL_ENABLE_IOSTREAM` is enabled. Default: `64`.
* `alpaka_SYCL_PLATFORM_ONEAPI`: set to `ON` to enable the oneAPI hardware targets. Requires the activation of at least one Intel hardware target (see below).
* `alpaka_SYCL_PLATFORM_XILINX`: set to `ON` to enable the AMD/Xilinx hardware targets.

### Building for Intel CPUs

The following CMake flags can be set for Intel CPUs:

* `alpaka_SYCL_ONEAPI_CPU`: set to `ON` to enable compilation for Intel CPUs.
* `alpaka_SYCL_ONEAPI_CPU_ISA`: the Intel ISA to compile for. Look at the possible `--march` options listed in the output of `opencl-aot --help`. Default: `avx2`.

### Building for Intel FPGAs

* `alpaka_SYCL_ONEAPI_FPGA`: set to `ON` to enable compilation for Intel FPGAs.
* `alpaka_SYCL_ONEAPI_FPGA_MODE`: the Intel FPGA compilation mode. Valid values are `emulation`, `simulation` and `hardware` which correspond to the Intel high-level synthesis targets with the same name. Default: `emulation`.
* `alpaka_SYCL_ONEAPI_FPGA_BOARD`: the Intel FPGA board to compile for. Ignored in `emulation` mode but important for `simulation` and `hardware` modes. Valid values are `pac_a10` (Arria 10 GX), `pac_s10` (Stratix 10 SX / D5005) and `pac_s10_usm` (Stratix 10 SX / D5005 with restricted USM). Default: `pac_a10`.
* `alpaka_SYCL_ONEAPI_FPGA_BSP`: the Intel FPGA board support package (BSP). Valid values are `intel_a10gx_pac` (Arria 10 GX) and `intel_s10sx_pac` (Stratix 10 SX / D5005). It is also possible to supply the full path to the BSP here if `aoc` is unable to look this up by itself. Note that the BSP must be chosen according to the selected board (see previous bullet point).

### Building for Intel GPUs

* `alpaka_SYCL_ONEAPI_GPU`: set to `ON` to enable compilation for Intel GPUs.
* `alpaka_SYCL_ONEAPI_GPU_DEVICES`: semicolon-separated list of one or more Intel GPUs to compile for. Check the output of `ocloc compile --help` and look at the possible values for the `-device` argument for valid values to supply here. Default: `bdw`.

### Building for AMD/Xilinx FPGAs

* `alpaka_SYCL_XILINX_FPGA_MODE`: the AMD/Xilinx FPGA compilation mode. Valid values are `simulation` and `hardware`. `simulation` refers to AMD/Xilinx' "hardware emulation" synthesis target, `hardware` to the "hardware" synthesis target. Note that `emulation` (which would correspond to AMD/Xilinx' "software emulation") is currently disabled because of constraints imposed by the AMD/Xilinx SYCL compiler. Default: `simulation`.

## Standalone mode

### General

Using the SYCL back-end always requires the following flags:

* `-fsycl` (compiler and linker)
* `-fsycl-standard=2020` (compiler)

To enable device-side printing add the following compiler flags:

* `-DALPAKA_SYCL_IOSTREAM_ENABLED`: to enable device-side printing.
* `-DALPAKA_SYCL_IOSTREAM_KIB=<value>`: `<value>` (without the brackets) defines the kibibytes per block to be reserved for device-side printing. `<value>` cannot exceed the amount of shared memory per block.

### Building for Intel CPUs

1. `#include <alpaka/standalone/CpuSyclIntel.hpp>` in your C++ code.
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
  * `-fsycl-targets=spir64_gen` (compiler and linker): to enable GPU compilation. Note: If you are using multiple SYCL hardware targets (like CPU and GPU) separate them by comma here.
  * `-Xsycl-target-backend=spir64_gen "-device <list>"` (linker): to choose the Intel GPU(s) to compile for. Multiple devices can either be separated by comma or by supplying a range of devices. Refer to the output of `ocloc compile --help` and look for the `-device` flag for the possible values.

### Building for AMD/Xilinx FPGAs

1. `#include <alpaka/standalone/FpgaSyclXilinx.hpp>` in your C++ code.
2. Add the following flags:
  * `-fsycl-targets=fpga64_hls_hw_emu`: to compile for AMD/Xilinx' "hardware emulation" target. Mutually exclusive with `fpga64_hls_hw`.
  * `-fsycl-targets=fpga64_hls_hw`: to compile for AMD/Xilinx "hardware" target. Mutually exclusive with `fpga64_hls_hw_emu`.

## Using the SYCL back-end

### Choosing the Accelerator

In contrast to the other back-ends the SYCL back-end comes with multiple different accelerators which should be chosen according to your requirements:

* `alpaka::experimental::AccCpuSyclIntel` for targeting Intel CPUs. In contrast to the other CPU back-ends this will be using Intel's OpenCL implementation for CPUs under the hood.
* `alpaka::experimental::AccFpgaSyclIntel` for targeting Intel FPGAs.
* `alpaka::experimental::AccGpuSyclIntel` for targeting Intel GPUs. 
* `alpaka::experimental::AccFpgaSyclXilinx` for targeting AMD/Xilinx FPGAs.

These can be used interchangeably (some restrictions apply - see below) with the non-experimental alpaka accelerators to compile an existing alpaka code for SYCL-capable hardware.

### Restrictions

* The FPGA back-ends (both vendors) cannot be used together with the Intel CPU / GPU back-ends. This is because of the different compilation trajectory required for FPGAs and is unlikely to be fixed anytime soon. See [here](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/programming-interface/fpga-flow/why-is-fpga-compilation-different.html) for an explanation.
* The SYCL back-end currently does not support passing pointers as kernel parameters. Use alpaka's experimental accessors instead.
* The SYCL back-end does not have device-side random number generation.
* The SYCL back-end does not support the (deprecated) `alpaka::clock()` function call.
* Similar to the CUDA and HIP back-ends the SYCL back-end only supports up to three kernel dimensions.
* Some Intel GPUs do not support the `double` type for device code. alpaka will not check this.
* The FPGA back-ends do not support atomics. alpaka will not check this.