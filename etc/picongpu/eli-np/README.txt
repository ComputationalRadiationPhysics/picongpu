- install the Spack package manager

  git clone -c feature.manyFiles=true https://github.com/spack/spack.git ${HOME}/spack
  spack spec zlib

- copy the compiler and package configuration:

  cp ${HOME}/src/picongpu/etc/picongpu/eli-np/spack/packages.yaml ${HOME}/.spack/
  cp ${HOME}/src/picongpu/etc/picongpu/eli-np/spack/compilers.yaml ${HOME}/.spack/linux/

- copy (and change if needed) the profile:

  cp ${HOME}/src/picongpu/etc/picongpu/eli-np/gpu_v100_picongpu.profile.example ${HOME}/gpu_v100_picongpu.profile

- create a Spack environment and install gcc

  spack env create gcc111-env ${HOME}/src/picongpu/etc/picongpu/eli-np/spack/gcc111-env.yaml
  spack env activate gcc111-env

  spack concretize
  spack install
  spack env deactivate

  spack compiler find `spack location -i gcc@11.1.0`

- create a Spack environment and install picongpu dependencies

  spack env create picongpu-env ${HOME}/src/picongpu/etc/picongpu/eli-np/spack/picongpu-env.yaml
  spack env activate picongpu-env

  spack concretize
  spack install

- in case one needs to start over, the clean-up procedure is the following

  spack env deactivate
  spack -e picongpu-env uninstall --all
  spack env rm picongpu-env

- optionally, for a full reset, do

  rm -rf ${HOME}/spack/ ${HOME}/.spack/

