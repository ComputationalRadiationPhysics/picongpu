#!/usr/bin/env python3

# generate a reduced test matrix based on the N-wise testing model
# https://en.wikipedia.org/wiki/All-pairs_testing

from allpairspy import AllPairs
import argparse
import sys


parser = argparse.ArgumentParser(description='Generate tesing pairs')
parser.add_argument('-n', dest='n_pairs', default=1, action="store",
                    help='number of tuple elements')
parser.add_argument('--compact', dest='compact', action="store_true",
                    help='print compact form of the test matrix')
args = parser.parse_args()
n_pairs = int(args.n_pairs)

examples = []
for i in sys.stdin:
    examples.append(i.rstrip())


def get_version(tuple):
    if len(tuple) >= 2:
        return float(tuple[1])
    return 0


# lookup table with compiler name and the required base container suffix
image_dict = {
    "g++": ".base_gcc",
    "g++_nvcc": ".base_nvcc",
    "clang++_nvcc": ".base_clangCuda",
    "clang++": ".base_clang",
    "clang++_clangCuda": ".base_clangCuda",
    "clang++_hipcc": ".base_hipcc"
}


def get_base_image(compiler, backend):
    lookup_name = compiler[0]
    if len(compiler) == 3:
        lookup_name += "_" + compiler[2]
    img_name = image_dict[lookup_name]
    if backend[0] == "cuda":
        img_name += "_" + backend[0] + "_" + str(backend[1])

    return img_name


# filter invalid cominations
#
# filter based on the compatibility overview
# https://gist.github.com/ax3l/9489132
def is_valid_combination(row):
    n = len(row)

    if n >= 2:
        v_compiler = get_version(row[0])

        is_clang_cuda = True if len(row[0]) == 3 and \
            row[0][2] == "clangCuda" else False
        is_clang = True if row[0][0] == "clang++" or is_clang_cuda else False

        is_gnu = True if row[0][0] == "g++" else False

        is_nvcc = True if len(row[0]) == 3 and row[0][2] == "nvcc" else False
        is_cuda = True if row[1][0] == "cuda" else False
        v_cuda = get_version(row[1])

        # hipcc
        is_hipcc = True if len(row[0]) == 3 and row[0][2] == "hipcc" else False
        is_hip = True if row[1][0] == "hip" else False

        # CI nvcc image is not shipped with clang++
        # clang_cuda images can currently not be used because
        # the base image is setting -DALPAKA_CUDA_COMPILER=clang
        if is_nvcc and is_clang:
            return False

        # hipcc is only valid in one combination
        if is_hip and is_hipcc and is_clang and v_compiler == 12:
            return True
        elif is_hip or is_hipcc:
            return False

        # clang 12 is currently only shipped with the HIP container
        if is_clang and v_compiler == 12:
            return False

        # docker images for clang cuda do not support clang++-7
        # together with cuda-9.2
        if is_clang_cuda and v_compiler == 7 and v_cuda == 9.2:
            return False

        # CUDA compiler requires backed `cuda`
        if (is_nvcc or is_clang_cuda) and not is_cuda:
            return False

        # cpu only compiler can not handle the backend `cuda`
        if (not is_nvcc and not is_clang_cuda) and is_cuda:
            return False

        # clang cuda compatibility
        if is_clang_cuda:
            if not is_cuda:
                return False
            if v_cuda == 9.2 and v_compiler >= 7:
                return True
            if v_cuda == 10.0 and v_compiler >= 8:
                return True
            if v_cuda == 10.1 and v_compiler >= 9:
                return True

            return False

        # nvcc compatibility
        if is_cuda and is_nvcc:
            if is_gnu:
                # g++-5.5 is not compatible with CUDA
                # https://github.com/tensorflow/tensorflow/issues/10220
                if v_compiler == 5:
                    return False
                if v_cuda <= 10.1 and v_compiler <= 7:
                    return True
                if v_cuda == 10.2 and v_compiler <= 8:
                    return True
                if v_cuda == 11.0 and v_compiler <= 9:
                    return True
                if v_cuda >= 11.1 and v_compiler <= 10:
                    return True

            if is_clang:
                if v_cuda == 9.2 and v_compiler <= 5:
                    return True
                if 10.0 <= v_cuda and v_cuda <= 10.2 and v_compiler <= 8:
                    return True
                if v_cuda == 11.0 and v_compiler <= 9:
                    return True
                if v_cuda >= 11.1 and v_compiler <= 10:
                    return True

            return False

    return True


# compiler list
# tuple with two components (compiler name, version)
clang_compiers = [("clang++", 5.0), ("clang++", 6.0), ("clang++", 7),
                  ("clang++", 8), ("clang++", 9), ("clang++", 10),
                  ("clang++", 11), ("clang++", 12)]
gnu_compilers = [("g++", 5), ("g++", 6), ("g++", 7), ("g++", 8),
                 ("g++", 9), ("g++", 10)]
compilers = [
    clang_compiers,
    gnu_compilers
]

# generate clang cuda compiler list
# add third component with the device compiler name
cuda_clang_compilers = []
for i in clang_compiers:
    cuda_clang_compilers.append(i + ("clangCuda", ))
compilers.append(cuda_clang_compilers)

# nvcc compiler
cuda_nvcc_compilers = []
for i in clang_compiers:
    cuda_nvcc_compilers.append(i + ("nvcc", ))
for i in gnu_compilers:
    cuda_nvcc_compilers.append(i + ("nvcc", ))
compilers.append(cuda_nvcc_compilers)

# hipcc compiler
hip_clang_compilers = []
for i in clang_compiers:
    hip_clang_compilers.append(i + ("hipcc", ))
compilers.append(hip_clang_compilers)

# PIConGPU backend list
# tuple with two components (backend name, version)
# version is only required for the cuda backend
backends = [("cuda", 9.2),
            ("cuda", 10.0), ("cuda", 10.1), ("cuda", 10.2),
            ("cuda", 11.0), ("cuda", 11.1), ("cuda", 11.2),
            ("omp2b", ), ("serial", ),
            ("hip", )]
boost_libs = ["1.65.1", "1.66.0", "1.67.0", "1.68.0", "1.69.0",
              "1.70.0", "1.71.0", "1.72.0", "1.73.0", "1.74.0"]

rounds = 1
# activate looping over the compiler categories to minimize the test matrix
# a small test matrix for each compiler e.g. clang, nvcc, g++, clang,
# clangCuda is created
if n_pairs == 1:
    rounds = len(compilers)

for i in range(rounds):
    used_compilers = []
    if n_pairs == 1:
        used_compilers = compilers[i]
    else:
        for c in compilers:
            used_compilers += c

    parameters = [
        used_compilers,
        backends,
        boost_libs,
        examples
    ]

    for i, pairs in enumerate(
            AllPairs(parameters,
                     filter_func=is_valid_combination, n=n_pairs)):
        if args.compact:
            print("{:2d}: {}".format(i, pairs))
        else:
            compiler = pairs[0][0] + "-" + str(pairs[0][1])
            backend = pairs[1][0]
            boost_version = pairs[2]
            folder = pairs[3]
            v_cuda = get_version(pairs[1])
            v_cuda_str = "" if v_cuda == 0 else str(v_cuda)
            job_name = compiler + "_" + backend + v_cuda_str + "_boost" + \
                boost_version + "_" + folder.replace("/", ".")
            print(job_name + ":")
            print("  variables:")
            print("    PIC_TEST_CASE_FOLDER: '" + folder + "'")
            print("    PIC_BACKEND: '" + backend + "'")
            print("    BOOST_VERSION: '" + boost_version + "'")
            print("    CXX_VERSION: '" + compiler + "'")
            print("  before_script:")
            print("    - apt-get update -qq")
            print("    - apt-get install -y -qq libopenmpi-dev "
                  "openmpi-bin openssh-server")
            print("  extends: " + get_base_image(pairs[0], pairs[1]))
            print("")
