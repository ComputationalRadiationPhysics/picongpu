Spack install scripts and profiles
==================================
This folder contains the scripts for installing the picongpu dependencies using spack and corresponding profiles.

> All install scripts assume an existing gcc 12.2.0 compiler install available in spack.
>
> Check with `spack compiler list`.
> If you do not see gcc@12.2.0 listed as a spack compiler, install/configure gcc
> 12.2.0 in spack first!

Installing picongpu dependencies
--------------------------------
To install the dependencies of picongpu for `<hardware>` you need to source one of the `<hardware>_install.sh` scripts.
i.e.
```bash
    source <hardware>_install.sh
```
> replace `<hardware>`, including angled brackets, by the correct script variant name

With `<hardware>` indicating the following hardware:
- `cpu` dependencies for running picongpu on cpu
- `gpu_a30` dependencies for running on nvidia a30 GPUs
- `gpu_v100` dependencies for running on nvidia v100 GPUs

> Do not run the same script if an existing functioning install already exists!
>
> Running the same script twice may create duplicate package installs if the spack concretization of the spec changed.

If no errors occurred during the install process, all dependencies for standard picongpu simulations have been installed.

> ISAAC is not currently installed by this script.

Deriving scripts for other Nvidia GPUs
---------------------------------------------

Scripts for other nvidia hardware may be derived by setting `cuda_arch=` in the spack to the correct value for the gpu.

> see CUDA support of gpu, for example NVIDIA A30 PCIe -> CUDA: 8.0 => `cuda_arch=80`
