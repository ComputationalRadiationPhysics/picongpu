#
# Copyright 2023 Benjamin Worpitz, Erik Zenker, Axel Hübl, Jan Stephan, René Widera, Jeffrey Kelling, Andrea Bocci,
#                Bernhard Manfred Gruber, Aurora Perego
# SPDX-License-Identifier: MPL-2.0
#

include(CheckLanguage) # check for CUDA/HIP language support
include(CMakePrintHelpers) # for easier printing of variables and properties
include(CMakeDependentOption) # Make options depend on other options

#-------------------------------------------------------------------------------
# Options.

# Compiler options
macro(alpaka_compiler_option name description default)
    if(NOT DEFINED alpaka_${name})
        set(alpaka_${name} ${default} CACHE STRING "${description}")
        set_property(CACHE alpaka_${name} PROPERTY STRINGS "DEFAULT;ON;OFF")
    endif()
endmacro()

# Add append compiler flags to a variable or target
#
# This method is automatically documenting all compile flags added into the variables
# alpaka_COMPILER_OPTIONS_HOST, alpaka_COMPILER_OPTIONS_DEVICE.
#
# scope - which compiler is effected: DEVICE, HOST, or HOST_DEVICE
# type - type of 'name': var, list, or target
#        var: space separated list
#        list: is semicolon separated
# name - name of the variable or target
# ... - parameter to appended to the variable or target 'name'
function(alpaka_set_compiler_options scope type name)
    if(scope STREQUAL HOST)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL DEVICE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL HOST_DEVICE)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'scope' unknown, value must be 'HOST', 'DEVICE', or 'HOST_DEVICE'.")
    endif()
    if(type STREQUAL "list")
        set(${name} ${${name}} ${ARGN} PARENT_SCOPE)
    elseif(type STREQUAL "var")
        foreach(arg IN LISTS ARGN)
            set(tmp "${tmp} ${arg}")
        endforeach()
        set(${name} "${${name}} ${tmp}" PARENT_SCOPE)
    elseif(type STREQUAL "target")
        foreach(arg IN LISTS ARGN)
            target_compile_options(${name} INTERFACE ${arg})
        endforeach()
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'type=${type}' unknown, value must be 'list', 'var', or 'target'.")
    endif()
endfunction()

# HIP and platform selection and warning about unsupported features
option(alpaka_ACC_GPU_HIP_ENABLE "Enable the HIP back-end (all other back-ends must be disabled)" OFF)
option(alpaka_ACC_GPU_HIP_ONLY_MODE "Only back-ends using HIP can be enabled in this mode." OFF) # HIP only runs without other back-ends

option(alpaka_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU back-end" OFF)
option(alpaka_ACC_GPU_CUDA_ONLY_MODE "Only back-ends using CUDA can be enabled in this mode (This allows to mix alpaka code with native CUDA code)." OFF)

if(alpaka_ACC_GPU_CUDA_ONLY_MODE AND NOT alpaka_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "If alpaka_ACC_GPU_CUDA_ONLY_MODE is enabled, alpaka_ACC_GPU_CUDA_ENABLE has to be enabled as well.")
endif()
if(alpaka_ACC_GPU_HIP_ONLY_MODE AND NOT alpaka_ACC_GPU_HIP_ENABLE)
    message(FATAL_ERROR "If alpaka_ACC_GPU_HIP_ONLY_MODE is enabled, alpaka_ACC_GPU_HIP_ENABLE has to be enabled as well.")
endif()

option(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU back-end" OFF)
option(alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread back-end" OFF)
option(alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE "Enable the TBB CPU grid block back-end" OFF)
option(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block back-end" OFF)
option(alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread back-end" OFF)
option(alpaka_ACC_CPU_DISABLE_ATOMIC_REF "Disable boost::atomic_ref for CPU back-ends" OFF)
option(alpaka_ACC_SYCL_ENABLE "Enable the SYCL back-end" OFF)

# Unified compiler options
alpaka_compiler_option(FAST_MATH "Enable fast-math" DEFAULT)
alpaka_compiler_option(FTZ "Set flush to zero" DEFAULT)

if((alpaka_ACC_GPU_CUDA_ONLY_MODE OR alpaka_ACC_GPU_HIP_ONLY_MODE)
   AND
    (alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
    alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
    alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
    alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
    alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR
    alpaka_ACC_SYCL_ENABLE))
    if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
        message(FATAL_ERROR "If alpaka_ACC_GPU_CUDA_ONLY_MODE is enabled, only back-ends using CUDA can be enabled! This allows to mix alpaka code with native CUDA code. However, this prevents any non-CUDA back-ends from being enabled.")
    endif()
    if(alpaka_ACC_GPU_HIP_ONLY_MODE)
        message(FATAL_ERROR "If alpaka_ACC_GPU_HIP_ONLY_MODE is enabled, only back-ends using HIP can be enabled!")
    endif()
    set(_alpaka_FOUND FALSE)
endif()

# avoids CUDA+HIP conflict
if(alpaka_ACC_GPU_HIP_ENABLE AND alpaka_ACC_GPU_CUDA_ENABLE)
    message(FATAL_ERROR "CUDA and HIP can not be enabled both at the same time.")
endif()

# HIP is only supported on Linux
if(alpaka_ACC_GPU_HIP_ENABLE AND (MSVC OR WIN32))
    message(FATAL_ERROR "Optional alpaka dependency HIP can not be built on Windows!")
endif()

# Drop-down combo box in cmake-gui.
set(alpaka_DEBUG "0" CACHE STRING "Debug level")
set_property(CACHE alpaka_DEBUG PROPERTY STRINGS "0;1;2")

set(alpaka_CXX_STANDARD_DEFAULT "17")
# Check whether alpaka_CXX_STANDARD has already been defined as a non-cached variable.
if(DEFINED alpaka_CXX_STANDARD)
    set(alpaka_CXX_STANDARD_DEFAULT ${alpaka_CXX_STANDARD})
endif()

set(alpaka_CXX_STANDARD ${alpaka_CXX_STANDARD_DEFAULT} CACHE STRING "C++ standard version")
set_property(CACHE alpaka_CXX_STANDARD PROPERTY STRINGS "17;20")

if(NOT TARGET alpaka)
    add_library(alpaka INTERFACE)

    target_compile_features(alpaka INTERFACE cxx_std_${alpaka_CXX_STANDARD})

    add_library(alpaka::alpaka ALIAS alpaka)
endif()

set(alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB "47" CACHE STRING "Kibibytes (1024B) of memory to allocate for block shared memory for backends requiring static allocation (includes CPU_B_OMP2_T_SEQ, CPU_B_TBB_T_SEQ, CPU_B_SEQ_T_SEQ, SYCL)")
alpaka_compiler_option(RELOCATABLE_DEVICE_CODE "Enable relocatable device code for CUDA, HIP and SYCL devices" DEFAULT)

# Random number generators
option(alpaka_DISABLE_VENDOR_RNG "Disable the vendor specific random number generators (NVIDIA cuRAND, AMD rocRAND, Intel DPL)" OFF)
if(alpaka_DISABLE_VENDOR_RNG)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_DISABLE_VENDOR_RNG")
endif()

# Device side assert
option(alpaka_ASSERT_ACC_ENABLE "Enable device side asserts. In case  value is OFF device side asserts will be disabled even if NDEBUG is not defined." ON)
if(!alpaka_ASSERT_ACC_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_DISABLE_ASSERT_ACC")
endif()

#-------------------------------------------------------------------------------
# Debug output of common variables.
if(${alpaka_DEBUG} GREATER 1)
    cmake_print_variables(_alpaka_ROOT_DIR)
    cmake_print_variables(_alpaka_COMMON_FILE)
    cmake_print_variables(_alpaka_ADD_EXECUTABLE_FILE)
    cmake_print_variables(_alpaka_ADD_LIBRARY_FILE)
    cmake_print_variables(CMAKE_BUILD_TYPE)
endif()

#-------------------------------------------------------------------------------
# Check supported compilers.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    message(FATAL_ERROR "Clang versions < 9 are not supported!")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message(WARNING "The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)!")
endif()

#-------------------------------------------------------------------------------
# Compiler settings.

if(MSVC)
    # warning C4505: '__cudaUnregisterBinaryUtil': unreferenced local function has been removed
    if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
        target_compile_options(alpaka INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4505>")
    endif()
else()
    # For std::future we need to pass the correct pthread flag for the compiler and the linker:
    # https://github.com/alpaka-group/cupla/pull/128#issuecomment-545078917

    # Allow users to override the "-pthread" preference.
    if(NOT THREADS_PREFER_PTHREAD_FLAG)
        set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    endif()

    find_package(Threads REQUIRED)
    target_link_libraries(alpaka INTERFACE Threads::Threads)

    if(NOT APPLE)
        # librt: undefined reference to `clock_gettime'
        find_library(RT_LIBRARY rt)
        if(RT_LIBRARY)
            target_link_libraries(alpaka INTERFACE ${RT_LIBRARY})
        endif()
    endif()

    # Add debug optimization levels. CMake doesn't do this by default.
    # Note that -Og is the recommended gcc optimization level for debug mode but is equivalent to -O1 for clang (and its derivates).
    alpaka_set_compiler_options(HOST_DEVICE target alpaka "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CXX>>:SHELL:-Og>"
                                                          "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-Xcompiler -Og>"
                                                          "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang,AppleClang,IntelLLVM>>:SHELL:-O0>"
                                                          "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:SHELL:/Od>")

    target_link_options(alpaka INTERFACE "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:SHELL:-Og>"
                                         "$<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang,AppleClang,IntelLLVM>>:SHELL:-O0>")
endif()

#-------------------------------------------------------------------------------
# Find Boost.
set(_alpaka_BOOST_MIN_VER "1.74.0")

if(${alpaka_DEBUG} GREATER 1)
    SET(Boost_DEBUG ON)
    SET(Boost_DETAILED_FAILURE_MSG ON)
endif()

find_package(Boost ${_alpaka_BOOST_MIN_VER} REQUIRED
             OPTIONAL_COMPONENTS atomic)

target_link_libraries(alpaka INTERFACE Boost::headers)

if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE OR
   alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE OR
   alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE OR
   alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR
   alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)

    if(NOT alpaka_ACC_CPU_DISABLE_ATOMIC_REF)
        # Check for C++20 std::atomic_ref first
        if(${alpaka_CXX_STANDARD} VERSION_GREATER_EQUAL "20")
            try_compile(alpaka_HAS_STD_ATOMIC_REF # Result stored here
                        "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
                        SOURCES "${_alpaka_FEATURE_TESTS_DIR}/StdAtomicRef.cpp" # Source file
                        CXX_STANDARD 20
                        CXX_STANDARD_REQUIRED TRUE
                        CXX_EXTENSIONS FALSE)
            if(alpaka_HAS_STD_ATOMIC_REF AND (NOT alpaka_ACC_CPU_DISABLE_ATOMIC_REF))
                message(STATUS "std::atomic_ref<T> found")
                target_compile_definitions(alpaka INTERFACE ALPAKA_HAS_STD_ATOMIC_REF)
            else()
                message(STATUS "std::atomic_ref<T> NOT found")
            endif()
        endif()

        if(Boost_ATOMIC_FOUND AND (NOT alpaka_HAS_STD_ATOMIC_REF))
            message(STATUS "boost::atomic_ref<T> found")
            target_link_libraries(alpaka INTERFACE Boost::atomic)
        endif()
    endif()

    if(alpaka_ACC_CPU_DISABLE_ATOMIC_REF OR ((NOT alpaka_HAS_STD_ATOMIC_REF) AND (NOT Boost_ATOMIC_FOUND)))
        message(STATUS "atomic_ref<T> was not found or manually disabled. Falling back to lock-based CPU atomics.")
        target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_ATOMIC_ATOMICREF)
    endif()
endif()

if(${alpaka_DEBUG} GREATER 1)
    message(STATUS "Boost in:")
    cmake_print_variables(BOOST_ROOT)
    cmake_print_variables(BOOSTROOT)
    cmake_print_variables(BOOST_INCLUDEDIR)
    cmake_print_variables(BOOST_LIBRARYDIR)
    cmake_print_variables(Boost_NO_SYSTEM_PATHS)
    cmake_print_variables(Boost_ADDITIONAL_VERSIONS)
    cmake_print_variables(Boost_USE_MULTITHREADED)
    cmake_print_variables(Boost_USE_STATIC_LIBS)
    cmake_print_variables(Boost_USE_STATIC_RUNTIME)
    cmake_print_variables(Boost_USE_DEBUG_RUNTIME)
    cmake_print_variables(Boost_USE_DEBUG_PYTHON)
    cmake_print_variables(Boost_USE_STLPORT)
    cmake_print_variables(Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS)
    cmake_print_variables(Boost_COMPILER)
    cmake_print_variables(Boost_THREADAPI)
    cmake_print_variables(Boost_NAMESPACE)
    cmake_print_variables(Boost_DEBUG)
    cmake_print_variables(Boost_DETAILED_FAILURE_MSG)
    cmake_print_variables(Boost_REALPATH)
    cmake_print_variables(Boost_NO_BOOST_CMAKE)
    message(STATUS "Boost out:")
    cmake_print_variables(Boost_FOUND)
    cmake_print_variables(Boost_INCLUDE_DIRS)
    cmake_print_variables(Boost_LIBRARY_DIRS)
    cmake_print_variables(Boost_LIBRARIES)
    cmake_print_variables(Boost_CONTEXT_FOUND)
    cmake_print_variables(Boost_CONTEXT_LIBRARY)
    cmake_print_variables(Boost_SYSTEM_FOUND)
    cmake_print_variables(Boost_SYSTEM_LIBRARY)
    cmake_print_variables(Boost_THREAD_FOUND)
    cmake_print_variables(Boost_THREAD_LIBRARY)
    cmake_print_variables(Boost_ATOMIC_FOUND)
    cmake_print_variables(Boost_ATOMIC_LIBRARY)
    cmake_print_variables(Boost_CHRONO_FOUND)
    cmake_print_variables(Boost_CHRONO_LIBRARY)
    cmake_print_variables(Boost_DATE_TIME_FOUND)
    cmake_print_variables(Boost_DATE_TIME_LIBRARY)
    cmake_print_variables(Boost_VERSION)
    cmake_print_variables(Boost_LIB_VERSION)
    cmake_print_variables(Boost_MAJOR_VERSION)
    cmake_print_variables(Boost_MINOR_VERSION)
    cmake_print_variables(Boost_SUBMINOR_VERSION)
    cmake_print_variables(Boost_LIB_DIAGNOSTIC_DEFINITIONS)
    message(STATUS "Boost cached:")
    cmake_print_variables(Boost_INCLUDE_DIR)
    cmake_print_variables(Boost_LIBRARY_DIR)
endif()

#-------------------------------------------------------------------------------
# If available, use C++20 math constants. Otherwise, fall back to M_PI etc.
if(${alpaka_CXX_STANDARD} VERSION_LESS "20")
    set(alpaka_HAS_STD_MATH_CONSTANTS FALSE)
else()
    try_compile(alpaka_HAS_STD_MATH_CONSTANTS # Result stored here
                "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
                SOURCES "${_alpaka_FEATURE_TESTS_DIR}/MathConstants.cpp" # Source file
                CXX_STANDARD 20
                CXX_STANDARD_REQUIRED TRUE
                CXX_EXTENSIONS FALSE)
endif()

if(NOT alpaka_HAS_STD_MATH_CONSTANTS)
    message(STATUS "C++20 math constants not found. Falling back to non-standard constants.")
    # Enable non-standard constants for MSVC.
    target_compile_definitions(alpaka INTERFACE "$<$<OR:$<CXX_COMPILER_ID:MSVC>,$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>>:_USE_MATH_DEFINES>")
endif()

#-------------------------------------------------------------------------------
# Find TBB.
if(alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    find_package(TBB 2021.4.0.0 REQUIRED)
    target_link_libraries(alpaka INTERFACE TBB::tbb)
endif()

#-------------------------------------------------------------------------------
# Find OpenMP.
if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
    if(APPLE)
        # Starting from Xcode 14.3.1 our self-compiled OpenMP libraries are no longer visible by default. We need to specify a few paths first.
        find_package(OpenMP COMPONENTS CXX)
        if(NOT OpenMP_CXX_FOUND)
            execute_process(COMMAND brew --prefix libomp
                            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)

            find_package(OpenMP REQUIRED COMPONENTS CXX)
        endif()
        target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)
    else()
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)
        # shown with CMake 3.29 and cray clang 17
        # workaround: cmake is missing to add '-fopenmp' to the linker flags
        if(CMAKE_CXX_COMPILER_ID STREQUAL "CrayClang")
            target_link_libraries(alpaka INTERFACE -fopenmp)
        endif()
    endif()
endif()

#-------------------------------------------------------------------------------
# Find CUDA.
if(alpaka_ACC_GPU_CUDA_ENABLE)
    # Save the user-defined host compiler (if any)
    set(_alpaka_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})

    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            # Use user selected CMake CXX compiler or CMAKE_CUDA_HOST_COMPILER as cuda host compiler to avoid fallback to the default system CXX host compiler.
            # CMAKE_CUDA_HOST_COMPILER is reset by check_language(CUDA) therefore definitions passed by the user via -DCMAKE_CUDA_HOST_COMPILER are
            # ignored by CMake (looks like a CMake bug).
            if(_alpaka_CUDA_HOST_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER ${_alpaka_CUDA_HOST_COMPILER})
            elseif("$ENV{CUDAHOSTCXX}" STREQUAL "")
                set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
            endif()
        endif()

        # the CMake compiler detection of clang 17 and 18 as CUDA compiler is broken
        # the detection try to compile an empty file with the default C++ standard of clang, which is -std=gnu++17
        # but CUDA does not support the 128 bit float extension, therefore the test failes
        # more details: https://gitlab.kitware.com/cmake/cmake/-/issues/25861
        # this workaround disable the gnu extensions for the compiler detection
        # the bug is fixed in clang 19: https://github.com/llvm/llvm-project/issues/88695
        if("${CMAKE_CUDA_COMPILER}" MATCHES "clang*")
            # get compiler version without enable_language()
            execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -dumpversion
                   OUTPUT_VARIABLE _CLANG_CUDA_VERSION
                   RESULT_VARIABLE _CLANG_CUDA_VERSION_ERROR_CODE)

            if(NOT "${_CLANG_CUDA_VERSION_ERROR_CODE}" STREQUAL "0")
                message(FATAL_ERROR "running '${CMAKE_CUDA_COMPILER} -dumpversion' failed: ${_CLANG_CUDA_VERSION_ERROR_CODE}")
            endif()

            string(STRIP ${_CLANG_CUDA_VERSION} _CLANG_CUDA_VERSION)
            message(DEBUG "Workaround: manual checked Clang-CUDA version: ${_CLANG_CUDA_VERSION}")

            if(${_CLANG_CUDA_VERSION} VERSION_GREATER_EQUAL 17 AND ${_CLANG_CUDA_VERSION} VERSION_LESS 19)
                message(DEBUG "Workaround: apply -std=c++98 for clang as cuda compiler")
                set(_CMAKE_CUDA_FLAGS_BEFORE ${CMAKE_CUDA_FLAGS})
                # we need to use C++ 98 for the detection test, because from new, disabling the extension is ignored for C++ 98
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++98")
            endif()
        endif()

        enable_language(CUDA)

        if(DEFINED _CLANG_CUDA_VERSION)
            message(DEBUG "Workaround: reset variables for clang as cuda compiler -std=c++98 fix")
            # remove the flag compiler -std=c++98
            set(CMAKE_CUDA_FLAGS ${_CMAKE_CUDA_FLAGS_BEFORE})
            unset(_CMAKE_CUDA_FLAGS_BEFORE)
            unset(_CLANG_CUDA_VERSION)
            unset(_CLANG_CUDA_VERSION_ERROR_CODE)
        endif()

        find_package(CUDAToolkit REQUIRED)

        target_compile_features(alpaka INTERFACE cuda_std_${alpaka_CXX_STANDARD})

        alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)
        alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)
        alpaka_compiler_option(CUDA_EXPT_EXTENDED_LAMBDA "Enable experimental, extended host-device lambdas in CUDA with nvcc" ON)

        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
            message(STATUS "clang is used as CUDA compiler")

            if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 14.0)
                # clang-14 is the first version to fully support CUDA 11.x
                message(FATAL_ERROR "clang as CUDA compiler requires at least clang-14.")
            else()
                message(WARNING "If you are using CUDA 11.3 please note of the following issue: https://github.com/alpaka-group/alpaka/issues/1857")
            endif()

            if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
                message(FATAL_ERROR "Clang as a CUDA compiler does not support OpenMP 2!")
            endif()

            target_compile_options(alpaka INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Wno-unknown-cuda-version>")

            # This flag silences the warning produced by the Dummy.cpp files:
            # clang: warning: argument unused during compilation: '--cuda-gpu-arch=sm_XX'
            # This seems to be a false positive as all flags are 'unused' for an empty file.
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Qunused-arguments>")

            # Silences warnings that are produced by boost because clang is not correctly identified.
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Wno-unused-local-typedef>")

            if(alpaka_FAST_MATH STREQUAL ON)
                # -ffp-contract=fast enables the usage of FMA
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-ffast-math -ffp-contract=fast>")
            endif()

            if(alpaka_FTZ STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fcuda-flush-denormals-to-zero>")
            endif()

            if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcuda-ptxas=-v>")
            endif()

            if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-save-temps>")
            endif()

        elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            message(STATUS "nvcc is used as CUDA compiler")

            if(alpaka_CXX_STANDARD GREATER_EQUAL 20 AND CMAKE_VERSION VERSION_LESS "3.25.0")
                message(FATAL_ERROR "CMake 3.24 and older does not support C++20 for nvcc")
            endif()

            # nvcc sets no linux/__linux macros on OpenPOWER linux
            # nvidia bug id: 2448610
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
                    alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Dlinux>")
                endif()
            endif()

            if(alpaka_CUDA_EXPT_EXTENDED_LAMBDA STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>")
            endif()
            # This is mandatory because with C++17 many standard library functions we rely on are constexpr (std::min, std::multiplies, ...)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>")

            # CMake automatically sets '-g' in debug mode
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-G>" # -G overrides -lineinfo
                                                             "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-g -lineinfo>")

            if(alpaka_FAST_MATH STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--use_fast_math>")
            endif()

            if(alpaka_FTZ STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ftz=true>")
            elseif(alpaka_FTZ STREQUAL OFF)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ftz=false>")
            endif()

            if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas -v>")
            endif()

            if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
                if(NOT MSVC)
                    alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp>")

                    # See https://github.com/alpaka-group/alpaka/issues/1755
                    if((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND
                       (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13))
                       message(STATUS "clang >= 13 detected. Force-setting OpenMP to version 4.5.")
                       alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp-version=45>")
                    endif()
                else()
                    alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /openmp>")
                endif()
            endif()

            # Always add warning/error numbers which can be used for suppressions
            set(ALPAKA_CUDA_DISPLAY_ERROR_NUM "$<IF:$<VERSION_LESS:$<CUDA_COMPILER_VERSION>,11.2.0>,-Xcudafe=--display_error_number,--display-error-number>")
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${ALPAKA_CUDA_DISPLAY_ERROR_NUM}>")

            if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--keep>")
            endif()

            option(alpaka_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES is enabled source code will be inlined in ptx." OFF)
            if(alpaka_CUDA_SHOW_CODELINES)
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--source-in-ptx -lineinfo>")

                # This is shaky - We currently don't have a way of checking for the host compiler ID.
                # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
                if(NOT MSVC)
                    alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -rdynamic>")
                endif()
                set(alpaka_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
            endif()
        endif()

        # Use the Shared CUDA Runtime library by default
        if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
            set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
        endif()

        # Link the CUDA Runtime library
        if(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Shared")
            target_link_libraries(alpaka INTERFACE CUDA::cudart)
        elseif(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Static")
            target_link_libraries(alpaka INTERFACE CUDA::cudart_static)
        elseif(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "None")
            message(WARNING "Building alpaka applications with CMAKE_CUDA_RUNTIME_LIBRARY=None is not supported.")
        else()
            message(FATAL_ERROR "Invalid setting for CMAKE_CUDA_RUNTIME_LIBRARY.")
        endif()

        if(NOT alpaka_DISABLE_VENDOR_RNG)
            # Use cuRAND random number generators
            if(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Shared")
                target_link_libraries(alpaka INTERFACE CUDA::curand)
            elseif(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Static")
                target_link_libraries(alpaka INTERFACE CUDA::curand_static)
            elseif(CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "None")
                message(FATAL_ERROR "cuRAND requires the CUDA runtime library.")
            endif()
        endif()
    else()
        message(FATAL_ERROR "Optional alpaka dependency CUDA could not be found!")
    endif()
endif()

#-------------------------------------------------------------------------------
# Find HIP.
if(alpaka_ACC_GPU_HIP_ENABLE)

    check_language(HIP)

    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        find_package(hip REQUIRED)

        set(_alpaka_HIP_MIN_VER 5.1)
        set(_alpaka_HIP_MAX_VER 6.2)

        # construct hip version only with major and minor level
        # cannot use hip_VERSION because of the patch level
        # 6.0 is smaller than 6.0.1234, so _alpaka_HIP_MAX_VER would have to be defined with a large patch level or
        # the next minor level, e.g. 6.1, would have to be used.
        set(_hip_MAJOR_MINOR_VERSION "${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")

        if(${_hip_MAJOR_MINOR_VERSION} VERSION_LESS ${_alpaka_HIP_MIN_VER} OR ${_hip_MAJOR_MINOR_VERSION} VERSION_GREATER ${_alpaka_HIP_MAX_VER})
            message(WARNING "HIP ${_hip_MAJOR_MINOR_VERSION} is not official supported by alpaka. Supported versions: ${_alpaka_HIP_MIN_VER} - ${_alpaka_HIP_MAX_VER}")
        endif()

        # let the compiler find the HIP headers also when building host-only code
        target_include_directories(alpaka SYSTEM INTERFACE ${hip_INCLUDE_DIR})

        target_link_libraries(alpaka INTERFACE "$<$<LINK_LANGUAGE:CXX>:hip::host>")
        alpaka_set_compiler_options(HOST_DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CXX>:-D__HIP_PLATFORM_AMD__>")
        if(${_hip_MAJOR_MINOR_VERSION} VERSION_EQUAL "5.1")
            alpaka_set_compiler_options(HOST_DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CXX>:-D__HIP_PLATFORM_HCC__>")
        endif()

        alpaka_compiler_option(HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" OFF)
        if(alpaka_HIP_KEEP_FILES)
            alpaka_set_compiler_options(HOST_DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-save-temps>")
        endif()

        if(alpaka_FAST_MATH STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-ffast-math>")
        endif()

        if(NOT alpaka_DISABLE_VENDOR_RNG)
            # hiprand requires ROCm implementation of random numbers by rocrand
            # hip::hiprand is currently not expressing this dependency
            find_package(rocrand REQUIRED CONFIG
                    HINTS "${ROCM_ROOT_DIR}/rocrand"
                    HINTS "/opt/rocm/rocrand")
            if(rocrand_FOUND)
                target_link_libraries(alpaka INTERFACE roc::rocrand)
            else()
                MESSAGE(FATAL_ERROR "Could not find rocRAND (also searched in: ROCM_ROOT_DIR=${ROCM_ROOT_DIR}/rocrand).")
            endif()

            # HIP random numbers
            find_package(hiprand REQUIRED CONFIG
                    HINTS "${HIP_ROOT_DIR}/hiprand"
                    HINTS "/opt/rocm/hiprand")
            if(hiprand_FOUND)
                target_link_libraries(alpaka INTERFACE hip::hiprand)
            else()
                MESSAGE(FATAL_ERROR "Could not find hipRAND (also searched in: HIP_ROOT_DIR=${HIP_ROOT_DIR}/hiprand).")
            endif()
        endif()

        if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fgpu-rdc>")
            target_link_options(alpaka INTERFACE "$<$<LINK_LANGUAGE:HIP>:SHELL:-fgpu-rdc --hip-link>")
        elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-gpu-rdc>")
        endif()
    else()
        message(FATAL_ERROR "Optional alpaka dependency HIP could not be found!")
    endif()
endif() # HIP

#-------------------------------------------------------------------------------
# SYCL settings
if(alpaka_ACC_SYCL_ENABLE)
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
        # Possible oneAPI targets
        cmake_dependent_option(alpaka_SYCL_ONEAPI_CPU "Enable oneAPI CPU targets for the SYCL back-end" OFF "alpaka_ACC_SYCL_ENABLE" OFF)
        cmake_dependent_option(alpaka_SYCL_ONEAPI_FPGA "Enable oneAPI FPGA targets for the SYCL back-end" OFF "alpaka_ACC_SYCL_ENABLE" OFF)
        cmake_dependent_option(alpaka_SYCL_ONEAPI_GPU "Enable oneAPI GPU targets for the SYCL back-end" OFF "alpaka_ACC_SYCL_ENABLE" OFF)
        # Intel FPGA emulation / simulation
        if(alpaka_SYCL_ONEAPI_FPGA)
            set(alpaka_SYCL_ONEAPI_FPGA_MODE "emulation" CACHE STRING "Synthesis type for oneAPI FPGA targets")
            set_property(CACHE alpaka_SYCL_ONEAPI_FPGA_MODE PROPERTY STRINGS "emulation;simulation;hardware")
        endif()

        alpaka_set_compiler_options(HOST_DEVICE target alpaka "-fsycl")
        target_link_options(alpaka INTERFACE "-fsycl")
        alpaka_set_compiler_options(HOST_DEVICE target alpaka "-sycl-std=2020")

        #-----------------------------------------------------------------------------------------------------------------
        # Determine SYCL targets
        set(alpaka_SYCL_ONEAPI_CPU_TARGET "spir64_x86_64")
        set(alpaka_SYCL_ONEAPI_FPGA_TARGET "spir64_fpga")
        set(alpaka_SYCL_ONEAPI_GPU_TARGET ${alpaka_SYCL_ONEAPI_GPU_DEVICES})

        if(alpaka_SYCL_ONEAPI_CPU)
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_ONEAPI_CPU_TARGET})
        endif()

        if(alpaka_SYCL_ONEAPI_FPGA)
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_ONEAPI_FPGA_TARGET})
        endif()

        if(alpaka_SYCL_ONEAPI_GPU)
            list(APPEND alpaka_SYCL_TARGETS ${alpaka_SYCL_ONEAPI_GPU_TARGET})
        endif()

        if(NOT alpaka_SYCL_TARGETS)
            message(FATAL_ERROR "You must specify at least one oneAPI hardware target!")
        endif()

        list(JOIN alpaka_SYCL_TARGETS "," alpaka_SYCL_TARGETS_CONCAT)
        alpaka_set_compiler_options(HOST_DEVICE target alpaka "-fsycl-targets=${alpaka_SYCL_TARGETS_CONCAT}")
        target_link_options(alpaka INTERFACE "-fsycl-targets=${alpaka_SYCL_TARGETS_CONCAT}")

        #-----------------------------------------------------------------------------------------------------------------
        # Determine actual hardware to compile for
        if(alpaka_SYCL_ONEAPI_CPU)
            set(alpaka_SYCL_ONEAPI_CPU_ISA "avx2" CACHE STRING "Intel ISA to compile for")
            set_property(CACHE alpaka_SYCL_ONEAPI_CPU_ISA PROPERTY STRINGS "sse4.2;avx;avx2;avx512")

            target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_ONEAPI_CPU")
            target_link_options(alpaka INTERFACE "SHELL:-Xsycl-target-backend=${alpaka_SYCL_ONEAPI_CPU_TARGET} \"-march=${alpaka_SYCL_ONEAPI_CPU_ISA}\"")
        endif()

        if(alpaka_SYCL_ONEAPI_FPGA)
            target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_ONEAPI_FPGA")

            if(alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "emulation")
                target_compile_definitions(alpaka INTERFACE "ALPAKA_FPGA_EMULATION")
                alpaka_set_compiler_options(DEVICE target alpaka "-Xsemulator")
                target_link_options(alpaka INTERFACE "-Xsemulator")
            elseif(alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "simulation")
                alpaka_set_compiler_options(DEVICE target alpaka "-Xssimulation")
                target_link_options(alpaka INTERFACE "-Xssimulation")
            elseif(alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "hardware")
                alpaka_set_compiler_options(DEVICE target alpaka "-Xshardware")
                target_link_options(alpaka INTERFACE "-Xshardware")
            endif()

            if(NOT alpaka_SYCL_ONEAPI_FPGA_MODE STREQUAL "emulation")
                set(alpaka_SYCL_ONEAPI_FPGA_BOARD "pac_a10" CACHE STRING "Intel FPGA board to compile for")
                set_property(CACHE alpaka_SYCL_ONEAPI_FPGA_BOARD PROPERTY STRINGS "pac_a10;pac_s10;pac_s10_usm")

                set(alpaka_SYCL_ONEAPI_FPGA_BSP "intel_a10gx_pac" CACHE STRING "Path to or name of the Intel FPGA board support package")
                set_property(CACHE alpaka_SYCL_ONEAPI_FPGA_BSP PROPERTY STRINGS "intel_a10gx_pac;intel_s10sx_pac")
                target_link_options(alpaka INTERFACE "-Xsycl-target-backend=${alpaka_SYCL_ONEAPI_FPGA_TARGET} \"-board=${alpaka_SYCL_ONEAPI_FPGA_BSP}:${alpaka_SYCL_ONEAPI_FPGA_BOARD}\"")
            endif()
        endif()

        if(alpaka_SYCL_ONEAPI_GPU)
            # Create a drop-down list (in cmake-gui) of valid Intel GPU targets. On the command line the user can specifiy
            # additional targets, such as ranges: "Gen8-Gen12LP" or lists: "icllp;skl".
            set(alpaka_SYCL_ONEAPI_GPU_DEVICES "intel_gpu_pvc" CACHE STRING "Intel GPU devices / generations to compile for")
            set_property(CACHE alpaka_SYCL_ONEAPI_GPU_DEVICES
                        PROPERTY STRINGS "intel_gpu_pvc;intel_gpu_acm_g12;intel_gpu_acm_g11;intel_gpu_acm_g10;intel_gpu_dg1;intel_gpu_adl_n;intel_gpu_adl_p;intel_gpu_rpl_s;intel_gpu_adl_s;intel_gpu_rkl;intel_gpu_tgllp;intel_gpu_icllp;intel_gpu_cml;intel_gpu_aml;intel_gpu_whl;intel_gpu_glk;intel_gpu_apl;intel_gpu_cfl;intel_gpu_kbl;intel_gpu_skl;intel_gpu_bdw")
            # If the user has given us a list turn all ';' into ',' to pacify the Intel OpenCL compiler.
            string(REPLACE ";" "," alpaka_SYCL_ONEAPI_GPU_DEVICES "${alpaka_SYCL_ONEAPI_GPU_DEVICES}")

            target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_ONEAPI_GPU")
        endif()

        #-----------------------------------------------------------------------------------------------------------------
        # Generic SYCL options
        alpaka_set_compiler_options(DEVICE target alpaka "-fsycl-unnamed-lambda") # Compiler default but made explicit here

        if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka "-fsycl-rdc")
            target_link_options(alpaka INTERFACE "-fsycl-rdc")
        elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka "-fno-sycl-rdc")
            target_link_options(alpaka INTERFACE "-fno-sycl-rdc")
        endif()
    else()
        message(FATAL_ERROR "alpaka currently does not support SYCL implementations other than oneAPI: ${CMAKE_CXX_COMPILER_ID}.")
    endif()

    if(NOT alpaka_DISABLE_VENDOR_RNG)
        # Use oneDPL random number generators
        find_package(oneDPL REQUIRED)
        target_link_libraries(alpaka INTERFACE oneDPL)
    endif()
endif()

#-------------------------------------------------------------------------------
# alpaka.
if(alpaka_ACC_GPU_CUDA_ONLY_MODE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ONLY_MODE")
    message(STATUS alpaka_ACC_GPU_CUDA_ONLY_MODE)
endif()

if(alpaka_ACC_GPU_HIP_ONLY_MODE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ONLY_MODE")
    message(STATUS alpaka_ACC_GPU_HIP_ONLY_MODE)
endif()

if(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
endif()

if(alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
endif()
if(alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_TBB_T_SEQ_ENABLED)
endif()
if(alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
endif()
if(alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED")
    message(STATUS alpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
endif()
if(alpaka_ACC_GPU_CUDA_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_CUDA_ENABLED")
    message(STATUS alpaka_ACC_GPU_CUDA_ENABLED)
endif()
if(alpaka_ACC_GPU_HIP_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_GPU_HIP_ENABLED")
    message(STATUS alpaka_ACC_GPU_HIP_ENABLED)
endif()

if(alpaka_ACC_SYCL_ENABLE)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_ACC_SYCL_ENABLED")

    if(alpaka_SYCL_ONEAPI_CPU)
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_CPU")
    endif()
    if(alpaka_SYCL_ONEAPI_FPGA)
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_FPGA")
    endif()
    if(alpaka_SYCL_ONEAPI_GPU)
        target_compile_definitions(alpaka INTERFACE "ALPAKA_SYCL_TARGET_GPU")
    endif()

    message(STATUS alpaka_ACC_SYCL_ENABLED)
endif()

target_compile_definitions(alpaka INTERFACE "ALPAKA_DEBUG=${alpaka_DEBUG}")

target_compile_definitions(alpaka INTERFACE "ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB=${alpaka_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB}")

if(alpaka_CI)
    target_compile_definitions(alpaka INTERFACE "ALPAKA_CI")
endif()

#-------------------------------------------------------------------------------
# Target.
if(TARGET alpaka)

    if(${alpaka_DEBUG} GREATER 1)
        # Compile options.
        get_property(_alpaka_COMPILE_OPTIONS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_COMPILE_OPTIONS)
        cmake_print_variables(_alpaka_COMPILE_OPTIONS_PUBLIC)

        # Compile definitions
        get_property(_alpaka_COMPILE_DEFINITIONS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_COMPILE_DEFINITIONS)
        cmake_print_variables(_alpaka_COMPILE_DEFINITIONS_PUBLIC)

        # Include directories.
        get_property(_alpaka_INCLUDE_DIRECTORIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        cmake_print_variables(_alpaka_INCLUDE_DIRECTORIES_PUBLIC)
    endif()

    # the alpaka library itself
    # SYSTEM voids showing warnings produced by alpaka when used in user applications.
    if(BUILD_TESTING)
        target_include_directories(alpaka INTERFACE ${_alpaka_INCLUDE_DIRECTORY})
    else()
        target_include_directories(alpaka SYSTEM INTERFACE ${_alpaka_INCLUDE_DIRECTORY})
    endif()

    if(${alpaka_DEBUG} GREATER 1)
        # Link libraries.
        # There are no PUBLIC_LINK_FLAGS in CMAKE:
        # http://stackoverflow.com/questions/26850889/cmake-keeping-link-flags-of-internal-libs
        get_property(_alpaka_LINK_LIBRARIES_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_LIBRARIES)
        cmake_print_variables(_alpaka_LINK_LIBRARIES_PUBLIC)

        get_property(_alpaka_LINK_FLAGS_PUBLIC
                     TARGET alpaka
                     PROPERTY INTERFACE_LINK_OPTIONS)
        cmake_print_variables(_alpaka_LINK_FLAGS_PUBLIC)
    endif()
endif()

# Compile options summary
if(alpaka_COMPILER_OPTIONS_DEVICE OR alpaka_COMPILER_OPTIONS_DEVICE)
    message("")
    message("List of compiler flags added by alpaka")
    if(alpaka_COMPILER_OPTIONS_HOST)
        message("host compiler:")
        message("    ${alpaka_COMPILER_OPTIONS_HOST}")
    endif()
    if(alpaka_COMPILER_OPTIONS_DEVICE)
        message("device compiler:")
        message("    ${alpaka_COMPILER_OPTIONS_DEVICE}")
    endif()
    message("")
endif()

#-------------------------------------------------------------------------------
# Include mdspan

set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka")
set_property(CACHE alpaka_USE_MDSPAN PROPERTY STRINGS "SYSTEM;FETCH;OFF")

if (alpaka_USE_MDSPAN STREQUAL "SYSTEM")
    find_package(mdspan REQUIRED)
    target_link_libraries(alpaka INTERFACE std::mdspan)
    target_compile_definitions(alpaka INTERFACE ALPAKA_USE_MDSPAN)
elseif (alpaka_USE_MDSPAN STREQUAL "FETCH")
    include(FetchContent)
    FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG 973ef6415a6396e5f0a55cb4c99afd1d1d541681
    )
    # we don't use FetchContent_MakeAvailable(mdspan) since it would also install mdspan
    # see also: https://stackoverflow.com/questions/65527126/how-to-disable-installation-a-fetchcontent-dependency
    FetchContent_GetProperties(mdspan)
    if(NOT mdspan_POPULATED)
        FetchContent_Populate(mdspan)
        if(${CMAKE_VERSION} VERSION_LESS "3.25.0")
            add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL)
        else()
            add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL SYSTEM)
        endif()
    endif()
    if(${CMAKE_VERSION} VERSION_LESS "3.25.0")
        get_target_property(mdspan_include_dir std::mdspan INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(alpaka SYSTEM INTERFACE ${mdspan_include_dir})
    else()
        target_link_libraries(alpaka INTERFACE std::mdspan)
    endif()
    target_compile_definitions(alpaka INTERFACE ALPAKA_USE_MDSPAN)
elseif (alpaka_USE_MDSPAN STREQUAL "OFF")
else()
    message(FATAL_ERROR "Invalid option for alpaka_USE_MDSPAN")
endif()

if (NOT alpaka_USE_MDSPAN STREQUAL "OFF")
    if (MSVC AND (alpaka_CXX_STANDARD LESS 20))
        message(WARNING "std::mdspan on MSVC requires C++20. Please enable C++20 via alpaka_CXX_STANDARD. Use of std::mdspan has been disabled.")
        set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka" FORCE)
    endif ()

    if (alpaka_ACC_GPU_CUDA_ENABLE AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        # this issue actually only occurs when the host compiler (not the CXX compiler) is clang, but cmake does not let us query the host compiler id
        # see: https://gitlab.kitware.com/cmake/cmake/-/issues/20901
        message(WARNING "std::mdspan does not work with nvcc and clang as host compiler. Use of std::mdspan has been disabled.")
        set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka" FORCE)
    endif ()

    if (alpaka_ACC_GPU_CUDA_ENABLE AND (NOT alpaka_CUDA_EXPT_EXTENDED_LAMBDA STREQUAL ON))
        message(WARNING "std::mdspan requires nvcc's extended lambdas. Use of std::mdspan has been disabled.")
        set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka" FORCE)
    endif()
endif()
