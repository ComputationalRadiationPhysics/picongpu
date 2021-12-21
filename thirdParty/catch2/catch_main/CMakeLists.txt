#
# Copyright 2015-2020 Benjamin Worpitz, Axel Huebl, Rene Widera
#
# This file is copied from lpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

option(USE_INTERNAL_CATCH2 "Use internally shipped Catch2" ON)

if(USE_INTERNAL_CATCH2)
    message(STATUS "Catch2: Using INTERNAL version 2.11.0")
else()
    find_package(Catch2 2.11.0 CONFIG REQUIRED)
    set_target_properties(Catch2::Catch2 PROPERTIES IMPORTED_GLOBAL TRUE)
    message(STATUS "Catch2: Found version ${Catch2_VERSION}")
endif()

add_library(CatchMain src/CatchMain.cpp)
# target_compile_features(CatchMain PUBLIC cxx_std_14)  # min C++14
set_target_properties(CatchMain PROPERTIES
    FOLDER "test"
    CXX_STANDARD 14  # exactly C++14
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_compile_definitions(CatchMain PUBLIC "CATCH_CONFIG_FAST_COMPILE")
if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    # Workaround for STL atomic issue: https://forums.developer.nvidia.com/t/support-for-atomic-in-libstdc-missing/135403/2
    # still appears in NVHPC 20.7
    target_compile_definitions(CatchMain PUBLIC "__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1")
endif()
if(MSVC)
    target_compile_definitions(CatchMain PUBLIC "CATCH_CONFIG_WINDOWS_CRTDBG")
    target_compile_options(CatchMain PUBLIC "/bigobj")
endif()

if(USE_INTERNAL_CATCH2)
    target_include_directories(CatchMain SYSTEM PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../include)
else()
    target_include_directories(CatchMain SYSTEM PUBLIC
        $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
