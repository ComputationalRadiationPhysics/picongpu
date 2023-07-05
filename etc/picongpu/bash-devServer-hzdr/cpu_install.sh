spack install --reuse cmake@3.25.0 %gcc@12.2.0
spack load cmake@3.25.0 ^openssl certs=mozilla %gcc@12.2.0

echo "openpmd-api:"
spack install --reuse openpmd-api@0.14.5 %gcc@12.2.0 \
    ^adios2@2.8.3 \
    ^cmake@3.25.0 \
    ^hdf5@1.12.2 \
    ^openmpi@4.1.4 +atomics\
    ^python@3.10.4 \
    ^py-numpy@1.23.3

echo "boost:"
spack install --reuse boost@1.80.0 \
    +program_options \
    +atomic \
    ~python \
    %gcc@12.2.0

echo "pngwriter"
spack install --reuse pngwriter@0.7.0 %gcc@12.2.0
