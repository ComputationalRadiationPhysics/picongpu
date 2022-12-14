#!/usr/bin/env python3

# generate a reduced test matrix based on the N-wise testing model
# https://en.wikipedia.org/wiki/All-pairs_testing

from allpairspy import AllPairs
import argparse
import sys
import math

parser = argparse.ArgumentParser(description='Generate tesing pairs')
parser.add_argument('-n', dest='n_pairs', default=1, action="store",
                    help='number of tuple elements')
# Note: If a stage contains to less jobs the number of jobs per stage can be
# larger than the value configured by the user!
parser.add_argument('-j', dest='num_jobs_per_stage', default=sys.maxsize,
                    action="store", help='number of per stage')
parser.add_argument('--compact', dest='compact', action="store_true",
                    help='print compact form of the test matrix')
parser.add_argument('--limit_boost_versions', dest='limit_boost_versions',
                    action="store_true",
                    help='test only every second boost version')
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
        img_name += "_" + backend[0]
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
        v_hip = get_version(row[1])

        os_name = row[2][0] if n >= 3 else ""
        os_version = get_version(row[2]) if n >= 3 else 0

        if is_cuda and os_name == "ubuntu":
            if os_version < 20.04 and v_cuda >= 11.0:
                return False
            elif os_version > 18.04 and v_cuda < 11.0:
                return False

        if is_hip and os_name == "ubuntu" and os_version < 20.04:
            return False

        # CI nvcc image is not shipped with clang++
        # clang_cuda images can currently not be used because
        # the base image is setting -DALPAKA_CUDA_COMPILER=clang
        if is_nvcc and is_clang:
            return False

        # hipcc is only valid in one combination
        if is_hip:
            if is_hipcc and is_clang:
                if 4.3 <= v_hip < 5.0 and v_compiler == 13:
                    return True
                if 5.0 <= v_hip < 6.0 and v_compiler == 14:
                    return True
            return False

        # hipcc should not be used without the hip backend
        if is_hipcc:
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
            # clang cuda containers shipped without clang-11 and clang-12
            # https://codebase.helmholtz.cloud/crp/alpaka-group-container/-/issues/28#note_3118676
            if v_compiler == 11 or v_compiler == 12:
                return False
            # native CMake CUDA compile is not supporting clang 8
            if v_compiler == 8:
                return False
            if v_cuda == 10.0 and v_compiler >= 8:
                return True
            if v_cuda == 10.1 and v_compiler >= 9:
                return True
            if 11.0 <= v_cuda <= 11.6 and v_compiler >= 11:
                return True

            return False

        # nvcc compatibility
        if is_cuda and is_nvcc:
            if is_gnu:
                # for C++17 support CUDA >= 11 is required
                if v_cuda == 11.0 and v_compiler <= 9:
                    return True
                if v_cuda >= 11.1 and v_compiler <= 10:
                    if v_compiler == 10:
                        # nvcc + gcc 10.3 bug see:
                        # https://github.com/alpaka-group/alpaka/issues/1297
                        return False
                    else:
                        return True

            if is_clang:
                # for C++17 support CUDA >= 11 is required
                if v_cuda == 11.0 and v_compiler <= 9:
                    return True
                if 11.1 <= v_cuda <= 11.6 and v_compiler <= 10:
                    return True

            return False

    return True


# compiler list
# tuple with two components (compiler name, version)
clang_compiers = [("clang++", 10), ("clang++", 11), ("clang++", 12),
                  ("clang++", 13), ("clang++", 14), ("clang++", 15)]
gnu_compilers = [("g++", 9), ("g++", 10), ("g++", 11)]
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
backends = [("hip", 5.0), ("hip", 5.1), ("hip", 5.2),
            ("cuda", 10.0), ("cuda", 10.1), ("cuda", 10.2),
            ("cuda", 11.0), ("cuda", 11.1), ("cuda", 11.2),
            ("cuda", 11.3), ("cuda", 11.4),
            ("omp2b", ), ("serial", )]

boost_libs_all = ["1.74.0", "1.75.0", "1.76.0",
                  "1.77.0", "1.78.0"]

operating_system = [("ubuntu", 18.04), ("ubuntu", 20.04)]

if args.limit_boost_versions:
    # select each second but keep the order
    boost_libs = boost_libs_all[-1::-2][::-1]
else:
    boost_libs = boost_libs_all

rounds = 1
# activate looping over the compiler categories to minimize the test matrix
# a small test matrix for each compiler e.g. clang, nvcc, g++, clang,
# clangCuda is created
if n_pairs == 1:
    rounds = len(compilers)


job_list = []

# generate a list with all jobs
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
        operating_system,
        boost_libs,
        examples
    ]

    for value in enumerate(
        AllPairs(parameters,
                 filter_func=is_valid_combination, n=n_pairs)):
        job_list.append(value)

num_jobs = len(job_list)
num_jobs_per_stage = int(args.num_jobs_per_stage)
num_stages = math.ceil(num_jobs / num_jobs_per_stage)

# generate stages
if not args.compact:
    print("stages:")
    for x in range(num_stages):
        print("  - job_{}".format(x))
    print("")

# generate all jobs
for stage in range(num_stages):
    if args.compact:
        print("---")
    for i, pairs in job_list[stage::num_stages]:
        if args.compact:
            print("{:2d}: {}".format(i, pairs))
        else:
            compiler = pairs[0][0] + "-" + str(pairs[0][1])
            backend = pairs[1][0]
            boost_version = pairs[3]
            folder = pairs[4]
            v_cuda_hip = get_version(pairs[1])
            v_cuda_hip_str = "" if v_cuda_hip == 0 else str(v_cuda_hip)
            os_name = pairs[2][0]
            os_version = get_version(pairs[2])
            image_prefix = "_run" if folder == "pmacc" else "_compile"
            job_name = compiler + "_" + backend + v_cuda_hip_str + \
                "_boost" + boost_version + "_" + folder.replace("/", ".")
            print(job_name + ":")
            print("  stage: job_{}".format(stage))
            print("  variables:")
            print("    CI_CONTAINER_NAME: '" + os_name + str(os_version) +
                  "'")
            if backend == "cuda":
                print("    CUDA_CONTAINER_VERSION: '" +
                      v_cuda_hip_str.replace('.', '') + "'")
            if backend == "hip":
                print("    HIP_CONTAINER_VERSION: '" +
                      v_cuda_hip_str + "'")
            print("    PIC_TEST_CASE_FOLDER: '" + folder + "'")
            print("    PIC_BACKEND: '" + backend + "'")
            print("    BOOST_VERSION: '" + boost_version + "'")
            print("    CXX_VERSION: '" + compiler + "'")
            print("  before_script:")
            if backend == "hip":
                print("    - wget -q -O - "
                      "https://repo.radeon.com/rocm/rocm.gpg.key | "
                      "apt-key add -")
            if backend == "cuda":
                print("    - apt-key adv --fetch-keys "
                      "https://developer.download.nvidia.com/compute"
                      "/cuda/repos/${CI_CONTAINER_NAME//.}"
                      "/x86_64/3bf863cc.pub")
            print("    - apt-get update -qq")
            print("    - apt-get install -y -qq libopenmpi-dev "
                  "openmpi-bin openssh-server")
            print("  extends: " + get_base_image(pairs[0], pairs[1]) +
                  image_prefix)
            print("")
