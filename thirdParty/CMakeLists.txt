#
# Copyright 2023 Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
# SPDX-License-Identifier: MPL-2.0
#

if(BUILD_TESTING OR alpaka_BUILD_BENCHMARKS)
    if(alpaka_USE_INTERNAL_CATCH2)
        message(STATUS "Catch2: Using INTERNAL version 3.5.2")
        # Force Catch2's CMake to pick up the variables we set below
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        set(CATCH_CONFIG_FAST_COMPILE OFF)
        if(WIN32)
            set(CATCH_CONFIG_WINDOWS_CRTDBG ON)
        endif()
        # Ensure that Catch2 and alpaka use the same C++ standard
        set(CMAKE_CXX_STANDARD ${alpaka_CXX_STANDARD})
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        set(CMAKE_CXX_EXTENSIONS OFF)
        # Workaround for STL atomic issue: https://forums.developer.nvidia.com/t/support-for-atomic-in-libstdc-missing/135403/2
        # still appears in NVHPC 20.7
        add_compile_definitions($<$<CXX_COMPILER_ID:PGI>:__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1>)

        # Silence icpx remark:
        # Note that use of '-g' without any optimization-level option will turn off most compiler optimizations similar to use of '-O0'
        # Remarks are not treated as warnings so we need to disable them manually. 
        add_compile_options($<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:IntelLLVM>>:-Rno-debug-disables-optimization>)

        # Turn off fast math to prevent warnings inside Catch2 (icpx defaults to -fp-model=fast)
        add_compile_options($<$<CXX_COMPILER_ID:IntelLLVM>:-fp-model=precise>)
        
        add_subdirectory(catch2)

        # hide Catch2 cmake variables by default in cmake gui
        get_cmake_property(variables VARIABLES)
        foreach (var ${variables})
            if (var MATCHES "^CATCH_")
                mark_as_advanced(${var})
            endif()
        endforeach()
    else()
        find_package(Catch2 3.5.2 CONFIG REQUIRED)
        message(STATUS "Catch2: Found version ${Catch2_VERSION}")
    endif()
endif()
