spack install --reuse cmake@3.26.3 %gcc@12.1.0
spack load cmake@3.26.3 %gcc@12.1.0

echo "openpmd-api:"
spack install --reuse openpmd-api@0.15.2 %gcc@12.1.0 \
    ^adios2@2.9.2 +cuda cuda_arch=80\
    ^cmake@3.26.3 \
    ^hdf5@1.14.0 \
    ^openmpi@4.1.5 +atomics +cuda cuda_arch=80\
    ^python@3.10.10 \
    ^py-numpy@1.24.2

echo "boost:"
spack install --reuse boost@1.81.0 \
    +program_options \
    +atomic \
    ~python \
    %gcc@12.1.0

echo "pngwriter"
spack install --reuse pngwriter@0.7.0 %gcc@12.1.0
