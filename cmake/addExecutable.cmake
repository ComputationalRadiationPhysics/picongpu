#
# Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

#------------------------------------------------------------------------------
#
# alpaka_add_executable(<name> [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [<source>...])
#
# Calls add_executable under the hood. Depending on the enabled back-ends, source file or target properties which
# cannot be propagated by the alpaka::alpaka target are set here.
#
# Using a macro to stay in the scope (fixes lost assignment of linker command in FindHIP.cmake)
# https://github.com/ROCm-Developer-Tools/HIP/issues/631

macro(alpaka_add_executable In_Name)

    add_executable(${In_Name} ${ARGN})

    if(alpaka_ACC_GPU_CUDA_ENABLE)
       enable_language(CUDA)
       foreach(_file ${ARGN})
            if((${_file} MATCHES "\\.cpp$") OR
               (${_file} MATCHES "\\.cxx$") OR
               (${_file} MATCHES "\\.cu$")
            )
                set_source_files_properties(${_file} PROPERTIES LANGUAGE CUDA)
            endif()
        endforeach()

        # We have to set this here since CUDA_SEPARABLE_COMPILATION is not propagated by the alpaka::alpaka target.
        if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
            set_property(TARGET ${In_Name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
            set_property(TARGET ${In_Name} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
        endif()
    endif()

    if(alpaka_ACC_GPU_HIP_ENABLE)
        enable_language(HIP)
        foreach(_file ${ARGN})
            if((${_file} MATCHES "\\.cpp$") OR
               (${_file} MATCHES "\\.cxx$") OR
               (${_file} MATCHES "\\.hip$")
            )
                set_source_files_properties(${_file} PROPERTIES LANGUAGE HIP)
            endif()
        endforeach()

        # We have to set this here because CMake currently doesn't provide hip_std_${VERSION} for
        # target_compile_features() and HIP_STANDARD isn't propagated by interface libraries.
        set_target_properties(${In_Name} PROPERTIES
                              HIP_STANDARD ${alpaka_CXX_STANDARD}
                              HIP_STANDARD_REQUIRED ON)
    endif()
endmacro()
