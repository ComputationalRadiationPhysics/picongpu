#spack install cmake@3.23.1 %gcc@10.2.0
spack load cmake@3.23.1 %gcc@10.2.0

echo "openpmd-api:"
spack install --reuse openpmd-api@0.14.4 %gcc@10.2.0 \
    ^adios2@2.8.0 \
    ^cmake@3.23.1 \
    ^hdf5@1.12.1 \
    ^openmpi@4.1.3 +atomics\
    ^python@3.9.12 \
    ^py-numpy@1.22.3

echo "boost:"
spack install --reuse boost@1.72.0 \
    +program_options \
    +filesystem \
    +system \
    +math \
    +serialization \
    +fiber \
    +context \
    +thread \
    +chrono \
    +atomic \
    +date_time %gcc@10.2.0\
    ^python@3.9.12