/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Matthias Werner, Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/test/dim/TestDims.hpp>
#include <alpaka/test/idx/TestIdxs.hpp>

#include <iosfwd>
#include <tuple>
#include <type_traits>

// When compiling the tests with CUDA enabled (nvcc or native clang) on the CI infrastructure
// we have to dramatically reduce the number of tested combinations.
// Else the log length would be exceeded.
#if defined(ALPAKA_CI)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA                                                       \
        || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
#        define ALPAKA_CUDA_CI
#    endif
#endif

namespace alpaka::test
{
    //! The detail namespace is used to separate implementation details from user accessible code.
    namespace detail
    {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
        template<typename TDim, typename TIdx>
        using AccCpuSerialIfAvailableElseInt = AccCpuSerial<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccCpuSerialIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) && !defined(ALPAKA_CUDA_CI)
        template<typename TDim, typename TIdx>
        using AccCpuThreadsIfAvailableElseInt = AccCpuThreads<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccCpuThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
        template<typename TDim, typename TIdx>
        using AccCpuTbbIfAvailableElseInt = AccCpuTbbBlocks<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccCpuTbbIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
        template<typename TDim, typename TIdx>
        using AccCpuOmp2BlocksIfAvailableElseInt = AccCpuOmp2Blocks<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccCpuOmp2BlocksIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) && !defined(ALPAKA_CUDA_CI)
        template<typename TDim, typename TIdx>
        using AccCpuOmp2ThreadsIfAvailableElseInt = AccCpuOmp2Threads<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccCpuOmp2ThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED) && !defined(TEST_UNIT_KERNEL_KERNEL_STD_FUNCTION)                         \
    && !(BOOST_COMP_PGI && (defined TEST_UNIT_FENCE))                                                                 \
    && !(                                                                                                             \
        BOOST_COMP_GNUC                                                                                               \
        && (((BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(11, 0, 0)) /* tests excluded because of GCC10 Oacc / Omp5 target \
                                                                   symbol bug with multiple units */                  \
             && (defined(TEST_UNIT_BLOCK_SHARED) || defined(TEST_UNIT_BLOCK_SYNC) || defined(TEST_UNIT_WARP)          \
                 || defined(TEST_UNIT_INTRINSIC) || defined(TEST_UNIT_KERNEL) || defined(TEST_UNIT_MEM_VIEW)))        \
            || defined(TEST_UNIT_MATH) /* because of static const members */                                          \
            ))                                                                                                        \
    && !(                                                                                                             \
        !defined(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST)                                                                    \
        && (defined(TEST_UNIT_ATOMIC) /* clang nvptx atomic ICEs */                                                   \
            ))
        template<typename TDim, typename TIdx>
        using AccOmp5IfAvailableElseInt = AccOmp5<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccOmp5IfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED) && !(defined(TEST_UNIT_KERNEL_KERNEL_STD_FUNCTION))                       \
    && !(                                                                                                             \
        defined(TEST_UNIT_TIME) /*no clock in OpenACC*/                                                               \
        || defined(TEST_UNIT_FENCE) /*no mem fence in OpenACC*/)                                                      \
    && !(                                                                                                             \
        BOOST_COMP_GNUC                                                                                               \
        && (((BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(11, 0, 0)) /* tests excluded because of GCC10 Oacc / Omp5 target \
                                                                   symbol bug with multiple units */                  \
             && (defined(TEST_UNIT_BLOCK_SHARED) || defined(TEST_UNIT_WARP) || defined(TEST_UNIT_INTRINSIC)           \
                 || defined(TEST_UNIT_KERNEL) || defined(TEST_UNIT_MEM_VIEW)))                                        \
            || defined(TEST_UNIT_MATH) /* because of static const members */                                          \
            || defined(TEST_UNIT_MEM_BUF) /* actually works, but hangs when ran by ctest */                           \
            ))                                                                                                        \
    && !(                                                                                                             \
        BOOST_COMP_PGI                                                                                                \
        && ((!defined(ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST))                                                              \
            && (defined(TEST_UNIT_MATH) /* 21.7: nvc++ fmod no acc device info */                                     \
                )))
        template<typename TDim, typename TIdx>
        using AccOaccIfAvailableElseInt = AccOacc<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccOaccIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && (BOOST_LANG_CUDA || defined(ALPAKA_HOST_ONLY))
        template<typename TDim, typename TIdx>
        using AccGpuCudaRtIfAvailableElseInt = AccGpuCudaRt<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccGpuCudaRtIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && (BOOST_LANG_HIP || defined(ALPAKA_HOST_ONLY))
        template<typename TDim, typename TIdx>
        using AccGpuHipRtIfAvailableElseInt =
            typename std::conditional<std::is_same_v<TDim, DimInt<3u>> == false, AccGpuHipRt<TDim, TIdx>, int>::type;
#else
        template<typename TDim, typename TIdx>
        using AccGpuHipRtIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_TARGET_CPU)
        template<typename TDim, typename TIdx>
        using AccCpuSyclIntelIfAvailableElseInt = alpaka::experimental::AccCpuSyclIntel<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccCpuSyclIntelIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_TARGET_FPGA)
        template<typename TDim, typename TIdx>
        using AccFpgaSyclIntelIfAvailableElseInt = alpaka::experimental::AccFpgaSyclIntel<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccFpgaSyclIntelIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)
        template<typename TDim, typename TIdx>
        using AccFpgaSyclXilinxIfAvailableElseInt = alpaka::experimental::AccFpgaSyclXilinx<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccFpgaSyclXilinxIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI) && defined(ALPAKA_SYCL_TARGET_GPU)
        template<typename TDim, typename TIdx>
        using AccGpuSyclIntelIfAvailableElseInt = alpaka::experimental::AccGpuSyclIntel<TDim, TIdx>;
#else
        template<typename TDim, typename TIdx>
        using AccGpuSyclIntelIfAvailableElseInt = int;
#endif

        //! A vector containing all available accelerators and int's.
        template<typename TDim, typename TIdx>
        using EnabledAccsElseInt = std::tuple<
            AccCpuSerialIfAvailableElseInt<TDim, TIdx>,
            AccCpuThreadsIfAvailableElseInt<TDim, TIdx>,
            AccCpuTbbIfAvailableElseInt<TDim, TIdx>,
            AccCpuOmp2BlocksIfAvailableElseInt<TDim, TIdx>,
            AccCpuOmp2ThreadsIfAvailableElseInt<TDim, TIdx>,
            AccOmp5IfAvailableElseInt<TDim, TIdx>,
            AccOaccIfAvailableElseInt<TDim, TIdx>,
            AccGpuCudaRtIfAvailableElseInt<TDim, TIdx>,
            AccGpuHipRtIfAvailableElseInt<TDim, TIdx>,
            AccCpuSyclIntelIfAvailableElseInt<TDim, TIdx>,
            AccFpgaSyclIntelIfAvailableElseInt<TDim, TIdx>,
            AccFpgaSyclXilinxIfAvailableElseInt<TDim, TIdx>,
            AccGpuSyclIntelIfAvailableElseInt<TDim, TIdx>>;
    } // namespace detail

    //! A vector containing all available accelerators.
    template<typename TDim, typename TIdx>
    using EnabledAccs = typename meta::Filter<detail::EnabledAccsElseInt<TDim, TIdx>, std::is_class>;

    namespace detail
    {
        //! The accelerator name write wrapper.
        struct StreamOutAccName
        {
            template<typename TAcc>
            ALPAKA_FN_HOST auto operator()(std::ostream& os) -> void
            {
                os << getAccName<TAcc>();
                os << " ";
            }
        };
    } // namespace detail

    //! Writes the enabled accelerators to the given stream.
    template<typename TDim, typename TIdx>
    ALPAKA_FN_HOST auto writeEnabledAccs(std::ostream& os) -> void
    {
        os << "Accelerators enabled: ";

        meta::forEachType<EnabledAccs<TDim, TIdx>>(detail::StreamOutAccName(), std::ref(os));

        os << std::endl;
    }

    namespace detail
    {
        //! A std::tuple holding multiple std::tuple consisting of a dimension and a idx type.
        //!
        //! TestDimIdxTuples =
        //!     tuple<
        //!         tuple<Dim1,Idx1>,
        //!         tuple<Dim2,Idx1>,
        //!         tuple<Dim3,Idx1>,
        //!         ...,
        //!         tuple<DimN,IdxN>>
        using TestDimIdxTuples = meta::CartesianProduct<std::tuple, TestDims, TestIdxs>;

        template<typename TList>
        using ApplyEnabledAccs = meta::Apply<TList, EnabledAccs>;

        //! A std::tuple containing std::tuple with fully instantiated accelerators.
        //!
        //! TestEnabledAccs =
        //!     tuple<
        //!         tuple<Acc1<Dim1,Idx1>, ..., AccN<Dim1,Idx1>>,
        //!         tuple<Acc1<Dim2,Idx1>, ..., AccN<Dim2,Idx1>>,
        //!         ...,
        //!         tuple<Acc1<DimN,IdxN>, ..., AccN<DimN,IdxN>>>
        using InstantiatedEnabledAccs = meta::Transform<TestDimIdxTuples, ApplyEnabledAccs>;
    } // namespace detail

    //! A std::tuple containing fully instantiated accelerators.
    //!
    //! TestAccs =
    //!     tuple<
    //!         Acc1<Dim1,Idx1>, ..., AccN<Dim1,Idx1>,
    //!         Acc1<Dim2,Idx1>, ..., AccN<Dim2,Idx1>,
    //!         ...,
    //!         Acc1<DimN,IdxN>, ..., AccN<DimN,IdxN>>
    using TestAccs = meta::Apply<detail::InstantiatedEnabledAccs, meta::Concatenate>;
} // namespace alpaka::test
