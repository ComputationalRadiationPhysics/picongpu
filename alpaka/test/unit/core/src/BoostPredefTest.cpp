/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>

TEST_CASE("printDefines", "[core]")
{
#if BOOST_LANG_CUDA
    std::cout << "BOOST_LANG_CUDA:" << BOOST_LANG_CUDA << std::endl;
#endif
#if BOOST_LANG_HIP
    std::cout << "BOOST_LANG_HIP:" << BOOST_LANG_HIP << std::endl;
#endif
#if BOOST_ARCH_PTX
    std::cout << "BOOST_ARCH_PTX:" << BOOST_ARCH_PTX << std::endl;
#endif
#if BOOST_ARCH_HSA
    std::cout << "BOOST_ARCH_HSA:" << BOOST_ARCH_HSA << std::endl;
#endif
#if BOOST_COMP_NVCC
    std::cout << "BOOST_COMP_NVCC:" << BOOST_COMP_NVCC << std::endl;
#endif
#if BOOST_COMP_HIP
    std::cout << "BOOST_COMP_HIP:" << BOOST_COMP_HIP << std::endl;
#endif
#if BOOST_COMP_CLANG
    std::cout << "BOOST_COMP_CLANG:" << BOOST_COMP_CLANG << std::endl;
#endif
#if BOOST_COMP_GNUC
    std::cout << "BOOST_COMP_GNUC:" << BOOST_COMP_GNUC << std::endl;
#endif
#if BOOST_COMP_INTEL
    std::cout << "BOOST_COMP_INTEL:" << BOOST_COMP_INTEL << std::endl;
#endif
#if BOOST_COMP_MSVC
    std::cout << "BOOST_COMP_MSVC:" << BOOST_COMP_MSVC << std::endl;
#endif
#if defined(BOOST_COMP_MSVC_EMULATED)
    std::cout << "BOOST_COMP_MSVC_EMULATED:" << BOOST_COMP_MSVC_EMULATED << std::endl;
#endif
#if BOOST_COMP_CLANG_CUDA
    std::cout << "BOOST_COMP_CLANG_CUDA:" << BOOST_COMP_CLANG_CUDA << std::endl;
#endif
#if BOOST_LIB_STD_GNU
    std::cout << "BOOST_LIB_STD_GNU:" << BOOST_LIB_STD_GNU << std::endl;
#endif
#if BOOST_LIB_STD_CXX
    std::cout << "BOOST_LIB_STD_CXX:" << BOOST_LIB_STD_CXX << std::endl;
#endif
#if BOOST_LIB_STD_DINKUMWARE
    std::cout << "BOOST_LIB_STD_DINKUMWARE:" << BOOST_LIB_STD_DINKUMWARE << std::endl;
#endif
}
