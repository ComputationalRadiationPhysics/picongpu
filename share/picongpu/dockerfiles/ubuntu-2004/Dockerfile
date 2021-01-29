FROM       nvidia/cuda:11.2.0-base-ubuntu20.04
MAINTAINER Axel Huebl <a.huebl@hzdr.de>
LABEL      authors="Axel Huebl, René Widera"

# docker and image environment
ENV        DEBIAN_FRONTEND=noninteractive \
           FORCE_UNSAFE_CONFIGURE=1 \
           SPACK_ROOT=/usr/local \
           SPACK_EXTRA_REPO=/usr/local/share/spack-repo \
           PIC_PACKAGE='picongpu@develop+isaac backend=cuda target=x86_64' \
           CUDA_PKG_VERSION="11-2"

# install minimal spack dependencies
#   - adds gfortran for spack's openmpi package
#   - adds the standard editors for users: vim, nano
#     (excluded: emacs-nox pulls up to 90MB in)
#   - adds build tools which are NOT linked in the final app
#   - selected cuda toolkit components
# note: do you need more cuda toolkit components?
# check:   apt-get update && apt-cache search cuda && apt-cache search nvcc
RUN        apt-get update && \
           apt-get install -y --no-install-recommends \
              autoconf \
              build-essential \
              ca-certificates \
              coreutils \
              cuda-cupti-$CUDA_PKG_VERSION \
              cuda-command-line-tools-$CUDA_PKG_VERSION \
              cuda-cudart-dev-$CUDA_PKG_VERSION \
              libcurand-dev-$CUDA_PKG_VERSION \
              cuda-minimal-build-$CUDA_PKG_VERSION \
              cuda-nvml-dev-$CUDA_PKG_VERSION \
              curl \
              environment-modules \
              gfortran \
              git \
              nano \
              openssh-client \
              pkg-config \
              python \
              rsync \
              time \
              unzip \
              vim && \
           rm -rf /var/lib/apt/lists/*

# spack settings
COPY       compilers.yaml packages.yaml modules.yaml $SPACK_ROOT/etc/spack/

# install spack && PIConGPU dependencies
# TODO: fix to a specific spack SHA or tag
RUN        curl -s -L https://github.com/spack/spack/archive/develop.tar.gz \
                | tar xzC $SPACK_ROOT --strip 1 && \
           mkdir -p $SPACK_EXTRA_REPO && \
           curl -s -L https://api.github.com/repos/ComputationalRadiationPhysics/spack-repo/tarball \
                | tar xzC $SPACK_EXTRA_REPO --strip 1 && \
           spack repo add --scope=system $SPACK_EXTRA_REPO
RUN        spack install --only dependencies $PIC_PACKAGE
RUN        spack install $PIC_PACKAGE && \
           spack clean -a

# load spack & picongpu environment on login
RUN        /bin/echo -e "source $SPACK_ROOT/share/spack/setup-env.sh\n" \
                        "spack load -r $PIC_PACKAGE\n" \
                        'if [ $(id -u) -eq 0 ]; then\n' \
                        '   function mpirun { $(which mpirun) --allow-run-as-root $@; }\n' \
                        '   export -f mpirun\n' \
                        'fi\n' \
                        'if [ $(id -u) -eq 0 ]; then\n' \
                        '   function mpiexec { $(which mpiexec) --allow-run-as-root $@; }\n' \
                        '   export -f mpiexec\n' \
                        'fi\n' \
               > /etc/profile.d/picongpu.sh

# force the use of a login shell
RUN        /bin/echo -e '#!/bin/bash -l\n' \
                        'exec "$@"\n' \
               > /etc/entrypoint.sh && \
           chmod a+x /etc/entrypoint.sh

# build example for out-of-the-box usage: LWFA
RUN        /bin/bash -l -c ' \
               pic-create $PICSRC/share/picongpu/examples/LaserWakefield /opt/picInputs/lwfa && \
               cd /opt/picInputs/lwfa && \
               pic-build -b "cuda:35;37;50;60;70;80" -c"-DCUDAMEMTEST_ENABLE=OFF" && \
               rm -rf .build && \
               chmod a+x /opt/picInputs/*/bin/* && \
               chmod a+r -R /opt/picInputs/* && \
               find /opt/picInputs -type d -exec chmod a+rx {} \;'
# KHI (Benchmark)
RUN        /bin/bash -l -c ' \
               pic-create $PICSRC/share/picongpu/examples/KelvinHelmholtz /opt/picInputs/khi && \
               cd /opt/picInputs/khi && \
               pic-build -b "cuda:35;37;50;60;70;80" -c"-DCUDAMEMTEST_ENABLE=OFF" && \
               rm -rf .build && \
               chmod a+x /opt/picInputs/*/bin/* && \
               chmod a+r -R /opt/picInputs/* && \
               find /opt/picInputs -type d -exec chmod a+rx {} \;'
# Laser-Ion Acceleration
RUN        /bin/bash -l -c ' \
               pic-create $PICSRC/share/picongpu/examples/FoilLCT /opt/picInputs/foil && \
               cd /opt/picInputs/foil && \
               pic-build -b "cuda:35;37;50;60;70;80" -c"-DCUDAMEMTEST_ENABLE=OFF" && \
               rm -rf .build && \
               chmod a+x /opt/picInputs/*/bin/* && \
               chmod a+r -R /opt/picInputs/* && \
               find /opt/picInputs -type d -exec chmod a+rx {} \;'
# ... also made input directories readable and files executable for all users

COPY       start_lwfa.sh /usr/bin/lwfa
COPY       start_lwfa_4.sh /usr/bin/lwfa4
COPY       start_lwfa_8.sh /usr/bin/lwfa8
COPY       start_lwfa_live.sh /usr/bin/lwfa_live
COPY       start_lwfa_live_4.sh /usr/bin/lwfa_live4
COPY       start_lwfa_live_8.sh /usr/bin/lwfa_live8
COPY       start_khi_1.sh /usr/bin/bench1
COPY       start_khi_4.sh /usr/bin/bench4
COPY       start_khi_8.sh /usr/bin/bench8
COPY       start_foil_4.sh /usr/bin/foil4
COPY       start_foil_8.sh /usr/bin/foil8

ENTRYPOINT ["/etc/entrypoint.sh"]
CMD ["/bin/bash"]
