FROM       nvidia/cuda:8.0-devel
MAINTAINER Axel Huebl <a.huebl@hzdr.de>

# docker and image environment
ENV        DEBIAN_FRONTEND=noninteractive \
           PICONGPU_BRANCH="0.3.0" \
           HOME=/home/picongpu \
           PATH=/home/picongpu/src/spack/bin:$PATH

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
              gfortran \
              git \
              lmod \
              nano \
              openssh-server \
              pkg-config \
              python \
              rsync \
              sudo \
              unzip \
              vim && \
           rm -rf /var/lib/apt/lists/*

# Add non-privileged, but system docker user with sudo rights
RUN        groupadd -r picongpu -g 901 && \
           useradd -u 901 -r -m -g picongpu picongpu && \
           echo "picongpu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/picongpu && \
           chmod 0440 /etc/sudoers.d/picongpu
# this is also the user that executes all commands below
USER       picongpu
WORKDIR    $HOME

# PIConGPU & spack environment
COPY       picongpu.profile /etc/profile.d/picongpu.sh
COPY       packages.yaml modules.yaml .spack/
RUN        mkdir build src paramSets && \
           sudo chown picongpu:picongpu -R .spack

# install spack && PIConGPU dependencies
# todo: fix to a specific spack SHA or tag
RUN        git clone --depth 50 https://github.com/llnl/spack.git \
           $HOME/src/spack/ && \
           spack install cmake@3.7.2 && \
           spack install cuda && \
           spack install openmpi@2.1.1 && \
           spack install zlib@1.2.11 && \
           spack install boost@1.62.0 && \
           spack install pngwriter@0.6.0 && \
           spack install libsplash@1.6.0 ^hdf5~fortran && \
           spack install adios@1.10.0 && \
           spack install isaac@1.3.1 ^boost@1.61.0 && \
           spack install isaac-server@1.3.1 ^boost@1.61.0 && \
           spack purge --all

# get PIConGPU sources
RUN        git clone --depth 50 --branch $PICONGPU_BRANCH \
               https://github.com/ComputationalRadiationPhysics/picongpu.git \
               $HOME/src/picongpu/

# build example for out-of-the-box usage: LWFA
RUN        /bin/bash -l -c ' \
               pic-create $PICSRC/examples/LaserWakefield $HOME/paramSets/lwfa && \
               cd build && \
               pic-configure -a "cuda:20;35;37;50;60" $HOME/paramSets/lwfa && \
               make -j install && \
               rm -rf ../build/*'

COPY       start_lwfa.sh /usr/bin/start_lwfa
CMD        /bin/bash -l
