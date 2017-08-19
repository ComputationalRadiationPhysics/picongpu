FROM       nvidia/cuda:8.0-devel
MAINTAINER Axel Huebl <a.huebl@hzdr.de>

# docker and image environment
ENV        DEBIAN_FRONTEND=noninteractive \
           FORCE_UNSAFE_CONFIGURE=1 \
           PICONGPU_BRANCH="dev" \
           SPACK_ROOT=/usr/local \
           PATH=/usr/local/bin:$PATH \
           PICSRC=/usr/local/src/picongpu

# install minimal spack dependencies
#   - adds gfortran for spack's openmpi package
#   - adds the standard editors for users: vim, nano, emacs
#   - adds build tools which are NOT linked in the final app
RUN        apt-get update && \
           apt-get install -y --no-install-recommends \
              autoconf \
              build-essential \
              ca-certificates \
              coreutils \
              curl \
              emacs \
              environment-modules \
              gfortran \
              git \
              nano \
              openssh-server \
              pkg-config \
              python \
              rsync \
              sudo \
              unzip \
              vim && \
           rm -rf /var/lib/apt/lists/*

# spack environment
COPY       picongpu.profile /etc/profile.d/picongpu.sh
COPY       packages.yaml modules.yaml /usr/local/etc/spack/

# install spack && PIConGPU dependencies
# todo: fix to a specific spack SHA or tag
RUN        curl -s -L https://api.github.com/repos/spack/spack/tarball \
                | tar xzC $SPACK_ROOT --strip 1 && \
           spack install cmake@3.7.2 && \
           spack install cuda@8.0.61 && \
           spack install openmpi@2.1.1 && \
           spack install zlib@1.2.11 && \
           spack install boost@1.62.0 && \
           spack install pngwriter@0.6.0 ^cmake@3.7.2 && \
           spack install libsplash@1.6.0 ^hdf5~fortran ^cmake@3.7.2 && \
           spack install adios@1.10.0 && \
           spack install isaac@1.3.1 ^boost@1.62.0 ^cmake@3.7.2 && \
           spack install isaac-server@1.3.1 ^boost@1.61.0 ^cmake@3.7.2 && \
           spack clean -a

# get PIConGPU sources
RUN        git clone --depth 50 --branch $PICONGPU_BRANCH \
               https://github.com/ComputationalRadiationPhysics/picongpu.git \
               $PICSRC

# build example for out-of-the-box usage: LWFA
RUN        /bin/bash -l -c ' \
               pic-create $PICSRC/share/picongpu/examples/LaserWakefield /opt/picInputs/lwfa && \
               cd /opt/picInputs/lwfa && \
               pic-build -b "cuda:20;35;37;50;60" && \
               rm -rf .build'

COPY       start_lwfa.sh /usr/bin/start_lwfa
CMD        /bin/bash -l
