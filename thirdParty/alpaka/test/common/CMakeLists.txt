#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.22)

set(_COMMON_TARGET_NAME "common")
set(_COMMON_COMPILE_OPTIONS_FILE "devCompileOptions.cmake")

add_library(${_COMMON_TARGET_NAME} INTERFACE)

include(${_COMMON_COMPILE_OPTIONS_FILE})
message(STATUS "alpaka_DEV_COMPILE_OPTIONS: ${alpaka_DEV_COMPILE_OPTIONS}")
target_compile_options(${_COMMON_TARGET_NAME} INTERFACE ${alpaka_DEV_COMPILE_OPTIONS})

if(MSVC)
    # This function or variable may be unsafe. Consider using <safe_version> instead.
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:SHELL:/wd4996>"
                                                            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4996>")
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:SHELL:/bigobj>"
                                                            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /bigobj>")
endif()

if(alpaka_ACC_GPU_CUDA_ENABLE)
    # CUDA driver API is used by EventHostManualTrigger
    target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE CUDA::cuda_driver)
    target_compile_definitions(${_COMMON_TARGET_NAME} INTERFACE "CUDA_API_PER_THREAD_DEFAULT_STREAM")
endif()

target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE alpaka::alpaka)

# Prevent "unsafe buffer usage" warnings from clang >= 16
if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "16.0.0") OR
   (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "2023.2.0"))
    
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE "-Wno-unsafe-buffer-usage")

    # We have no way to determine if we are using amdclang++. So we will just decide this by checking for the HIP back-end.
    if(alpaka_ACC_GPU_HIP_ENABLE)
        # amdclang++-5.5 pretends to be clang-16 but doesn't know all warnings.
        target_compile_options(${_COMMON_TARGET_NAME} INTERFACE "-Wno-unknown-warning-option") 
    endif()
endif()

# Prevent warnings from shady code inside Catch2
get_target_property(alpaka_CATCH2_INCLUDE_DIRS Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(${_COMMON_TARGET_NAME} SYSTEM INTERFACE ${alpaka_CATCH2_INCLUDE_DIRS})

# Now we can safely link to Catch2
target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE Catch2::Catch2WithMain)

if(TARGET ${_COMMON_TARGET_NAME})
    # HACK: Workaround for the limitation that files added to INTERFACE targets (target_sources) can not be marked as PUBLIC or PRIVATE but only as INTERFACE.
    # Therefore those files will be added to projects "linking" to the INTERFACE library, but are not added to the project itself within an IDE.
    add_custom_target("${_COMMON_TARGET_NAME}Ide" SOURCES ${_COMMON_FILES_HEADER} ${_COMMON_COMPILE_OPTIONS_FILE})
    set_target_properties("${_COMMON_TARGET_NAME}Ide" PROPERTIES FOLDER "test")
endif()
