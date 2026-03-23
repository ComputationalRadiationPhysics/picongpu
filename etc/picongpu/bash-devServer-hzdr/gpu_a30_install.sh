echo "cmake:"
spack install --reuse cmake@3.30
spack load cmake@3.30 ^openssl certs=mozilla

echo "openpmd-api:"
spack install --reuse openpmd-api@0.15.2 +python \
    ^adios2@2.9.2 ++blosc2 +cuda cuda_arch=80 \
    ^cmake@3.30 \
    ^hdf5@1.14.3 \
    ^openmpi@4.1.5 +atomics +cuda cuda_arch=80 \
    ^python@3.11 \
    ^py-numpy@1.26

echo "boost:"
spack install --reuse boost@1.83.0 \
    +program_options \
    +atomic \
    ~python \
    cxxstd=20

echo "pngwriter"
spack install --reuse pngwriter@0.7.0

echo "pip:"
spack mark -e py-pip ^python@3.11
