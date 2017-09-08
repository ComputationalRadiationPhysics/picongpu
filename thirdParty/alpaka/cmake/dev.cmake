#
# Copyright 2014-2016 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

#-------------------------------------------------------------------------------
# Compiler settings.
#-------------------------------------------------------------------------------

#MSVC
IF(MSVC)
    # Force to always compile with W4 and WX
    LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "/W4")
    LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "/WX")
    # Improve debugging.
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-d2Zi+")
    ENDIF()
    IF(MSVC_VERSION GREATER 1900)
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "/permissive-")
        IF(MSVC_VERSION GREATER 1910)
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "/Zc:twoPhase-")
        ENDIF()
    ENDIF()
    IF(MSVC_VERSION GREATER 1800)
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "/Zc:throwingNew" "/Zc:strictStrings")
    ENDIF()
ELSE()
    # GNU
    IF(CMAKE_COMPILER_IS_GNUCXX)
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wall")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wextra")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-pedantic")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Werror")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wdouble-promotion")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wmissing-include-dirs")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wunknown-pragmas")
        # Higher levels (max is 5) produce some strange warnings
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wstrict-overflow=2")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wtrampolines")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wfloat-equal")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wundef")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wshadow")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wcast-qual")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wcast-align")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wwrite-strings")
        # Too noisy as it warns for every operation using numeric types smaller then int.
        # Such values are converted to int implicitly before the calculation is done.
        # E.g.: uint16_t = uint16_t * uint16_t will trigger the following warning:
        # conversion to ‘short unsigned int’ from ‘int’ may alter its value
        #LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wconversion")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wsign-conversion")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wvector-operation-performance")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wzero-as-null-pointer-constant")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wdate-time")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wuseless-cast")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wlogical-op")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-aggressive-loop-optimizations")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wmissing-declarations")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-multichar")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wopenmp-simd")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wpacked")
        # Too much noise
        #LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wpadded")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wredundant-decls")
        # Too much noise
        #LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Winline")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wdisabled-optimization")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wformat-nonliteral")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wformat-security")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wformat-y2k")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wctor-dtor-privacy")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wdelete-non-virtual-dtor")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wliteral-suffix")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wnon-virtual-dtor")
        # This warns about members that have not explicitly been listed in the constructor initializer list.
        # This could be useful even for members that have a default constructor.
        # However, it also issues this warning for defaulted constructurs.
        #LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Weffc++")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Woverloaded-virtual")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wsign-promo")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wconditionally-supported")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wnoexcept")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wold-style-cast")
        IF(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wsuggest-final-types")
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wsuggest-final-methods")
            # This does not work correctly as it suggests override to methods that are already marked with final.
            # Because final implies override, this is not useful.
            #LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wsuggest-override")
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wnormalized")
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wformat-signedness")
        ENDIF()
        IF(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wnull-dereference")
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wduplicated-cond")
            LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wsubobject-linkage")
        ENDIF()
        # By marking the boost headers as system headers, warnings produced within them are ignored.
        FIND_PACKAGE(Boost QUIET)
        IF(NOT Boost_FOUND)
            MESSAGE(FATAL_ERROR "Required alpaka dependency Boost.Test could not be found!")
        ENDIF()
        INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})

    # Clang or AppleClang
    ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Werror")
        # Weverything really means everything (including Wall, Wextra, pedantic, ...)
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Weverything")
        # We are not C++98 compatible (we use C++11 features)
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-c++98-compat")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-c++98-compat-pedantic")
        # The following warnings are triggered by all instantiations of BOOST_AUTO_TEST_SUITE
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-disabled-macro-expansion")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-global-constructors")
        # This padding warning is generated by the executors depending on the argument types
        # as they are stored as members. Therefore, the padding warning is triggered by the calling code
        # and does not indicate a failure within alpaka.
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wno-padded")
        # By marking the boost headers as system headers, warnings produced within them are ignored.
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "--system-header-prefix=boost/")
    # ICC
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Wall")
    # PGI
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
        LIST(APPEND ALPAKA_DEV_COMPILE_OPTIONS "-Minform=inform")
    ENDIF()
ENDIF()