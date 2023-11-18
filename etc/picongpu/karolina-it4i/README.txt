- install the Spack package manager

- copy the compiler and package configuration:

  cp ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/spack/packages.yaml ${HOME}/.spack/
  cp ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/spack/compilers.yaml ${HOME}/.spack/linux/

- copy (and change if needed) the profile:

  cp ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/gpu_a100_picongpu.profile.example ${HOME}/gpu_a100_picongpu.profile

- create a Spack environment and install picongpu dependencies

  spack env create picongpu ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/spack/spack.yaml
  spack env activate picongpu

  spack concretize
  spack install

