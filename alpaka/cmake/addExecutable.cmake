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
    endif()
endmacro()
