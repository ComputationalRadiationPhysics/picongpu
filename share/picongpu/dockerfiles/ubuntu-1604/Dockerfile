FROM       nvidia/cuda:9.0-base
MAINTAINER Axel Huebl <a.huebl@hzdr.de>

# docker and image environment
ENV        DEBIAN_FRONTEND=noninteractive \
           FORCE_UNSAFE_CONFIGURE=1 \
           SPACK_ROOT=/usr/local \
           SPACK_EXTRA_REPO=/usr/local/share/spack-repo \
           PIC_PACKAGE='picongpu@develop+isaac backend=cuda'

# install minimal spack dependencies
#   - adds gfortran for spack's openmpi package
#   - adds the standard editors for users: vim, nano
#     (excluded: emacs-nox pulls up to 90MB in)
#   - adds build tools which are NOT linked in the final app
#   - selected cuda toolkit components
# need more cuda toolkit components?
# check:   apt-get update && apt-cache search cuda && apt-cache search nvcc
RUN        apt-get update && \
           apt-get install -y --no-install-recommends \
              autoconf \
              build-essential \
              ca-certificates \
              coreutils \
              cuda-cublas-dev-$CUDA_PKG_VERSION \
              cuda-command-line-tools-$CUDA_PKG_VERSION \
              cuda-core-$CUDA_PKG_VERSION \
              cuda-cudart-dev-$CUDA_PKG_VERSION \
              cuda-curand-dev-$CUDA_PKG_VERSION \
              cuda-minimal-build-$CUDA_PKG_VERSION \
              cuda-misc-headers-$CUDA_PKG_VERSION \
              cuda-nvml-dev-$CUDA_PKG_VERSION \
              curl \
              environment-modules \
              gfortran \
              git \
              nano \
              openssh-server \
              pkg-config \
              python \
              rsync \
              unzip \
              vim && \
           rm -rf /var/lib/apt/lists/*

# spack settings
COPY       packages.yaml modules.yaml $SPACK_ROOT/etc/spack/

# install spack && PIConGPU dependencies
# TODO: fix to a specific spack SHA or tag
RUN        curl -s -L https://github.com/spack/spack/archive/develop.tar.gz \
                | tar xzC $SPACK_ROOT --strip 1 && \
           mkdir -p $SPACK_EXTRA_REPO && \
           curl -s -L https://api.github.com/repos/ComputationalRadiationPhysics/spack-repo/tarball \
                | tar xzC $SPACK_EXTRA_REPO --strip 1 && \
           spack repo add $SPACK_EXTRA_REPO
RUN        spack install $PIC_PACKAGE && \
           spack clean -a

# load spack & picongpu environment on login
RUN        /bin/echo -e "source $SPACK_ROOT/share/spack/setup-env.sh\n" \
                        "spack load $PIC_PACKAGE" \
               > /etc/profile.d/picongpu.sh

# build example for out-of-the-box usage: LWFA
RUN        /bin/bash -l -c ' \
               pic-create $PICSRC/share/picongpu/examples/LaserWakefield /opt/picInputs/lwfa && \
               cd /opt/picInputs/lwfa && \
               pic-build -b "cuda:30;35;37;50;60" && \
               rm -rf .build'

COPY       start_lwfa.sh /usr/bin/start_lwfa
CMD        /bin/bash -l
