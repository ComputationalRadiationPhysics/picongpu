- install the Spack package manager

  git clone -c feature.manyFiles=true https://github.com/spack/spack.git
  spack spec zlib

- copy the compiler and package configuration:

  cp ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/spack/packages.yaml ${HOME}/.spack/
  cp ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/spack/compilers.yaml ${HOME}/.spack/linux/

- copy (and change if needed) the profile:

  cp ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/gpu_a100_picongpu.profile.example ${HOME}/gpu_a100_picongpu.profile

- create a Spack environment and install picongpu dependencies

  spack env create picongpu-env ${HOME}/src/picongpu/etc/picongpu/karolina-it4i/spack/spack.yaml
  spack env activate picongpu-env

  spack concretize
  spack install

- in case one needs to start over, the clean-up procedure is the following

  despacktivate
  spack -e picongpu-env uninstall --all
  spack env rm picongpu-env

- optionally, for a full reset, do

  rm -rf ${HOME}/spack/ ${HOME}/.spack/

