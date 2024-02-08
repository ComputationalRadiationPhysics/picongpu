echo "cmake:"
spack install --reuse cmake@3.26.6 %gcc@12.2.0
spack load cmake@3.26.6 ^openssl certs=mozilla %gcc@12.2.0

echo "openpmd-api:"
spack install --reuse openpmd-api@0.15.2 +python %gcc@12.2.0 \
    ^adios2@2.9.2 ++blosc2 +cuda cuda_arch=80 \
    ^cmake@3.26.6 \
    ^hdf5@1.14.3 \
    ^openmpi@4.1.5 +atomics +cuda cuda_arch=80 \
    ^python@3.11.6 \
    ^py-numpy@1.26.2

echo "boost:"
spack install --reuse boost@1.83.0 \
    +program_options \
    +atomic \
    ~python \
    cxxstd=17 \
    %gcc@12.2.0

echo "pngwriter"
spack install --reuse pngwriter@0.7.0 %gcc@12.2.0

echo "pip:"
spack mark -e py-pip ^python@3.11.6 %gcc@12.2.0
