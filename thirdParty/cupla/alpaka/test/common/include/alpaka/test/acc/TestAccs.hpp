/* Copyright 2019 Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <alpaka/test/dim/TestDims.hpp>
#include <alpaka/test/idx/TestIdxs.hpp>

#include <tuple>
#include <type_traits>
#include <iosfwd>

// When compiling the tests with CUDA enabled (nvcc or native clang) on the CI infrastructure
// we have to dramatically reduce the number of tested combinations.
// Else the log length would be exceeded.
#if defined(ALPAKA_CI)
  #if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA \
   || defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
    #define ALPAKA_CUDA_CI
  #endif
#endif

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The test specifics.
    namespace test
    {
        //-----------------------------------------------------------------------------
        //! The test accelerator specifics.
        namespace acc
        {
            //-----------------------------------------------------------------------------
            //! The detail namespace is used to separate implementation details from user accessible code.
            namespace detail
            {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuSerialIfAvailableElseInt = alpaka::acc::AccCpuSerial<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuSerialIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuThreadsIfAvailableElseInt = alpaka::acc::AccCpuThreads<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuFibersIfAvailableElseInt = alpaka::acc::AccCpuFibers<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuFibersIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuTbbIfAvailableElseInt = alpaka::acc::AccCpuTbbBlocks<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuTbbIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2BlocksIfAvailableElseInt = alpaka::acc::AccCpuOmp2Blocks<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2BlocksIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2ThreadsIfAvailableElseInt = alpaka::acc::AccCpuOmp2Threads<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp2ThreadsIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_CPU_BT_OMP4_ENABLED) && !defined(ALPAKA_CUDA_CI)
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp4IfAvailableElseInt = alpaka::acc::AccCpuOmp4<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccCpuOmp4IfAvailableElseInt = int;
#endif
#if (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuUniformCudaHipRtIfAvailableElseInt = alpaka::acc::AccGpuUniformCudaHipRt<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuUniformCudaHipRtIfAvailableElseInt = int;
#endif

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuCudaRtIfAvailableElseInt = alpaka::acc::AccGpuCudaRt<TDim, TIdx>;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuCudaRtIfAvailableElseInt = int;
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuHipRtIfAvailableElseInt = typename
                    std::conditional<
                    std::is_same<TDim,alpaka::dim::DimInt<3u>>::value==false,
                    alpaka::acc::AccGpuHipRt<TDim, TIdx>,
                    int>::type;
#else
                template<
                    typename TDim,
                    typename TIdx>
                using AccGpuHipRtIfAvailableElseInt = int;
#endif
                //#############################################################################
                //! A vector containing all available accelerators and void's.
                template<
                    typename TDim,
                    typename TIdx>
                using EnabledAccsElseInt =
                    std::tuple<
                        AccCpuSerialIfAvailableElseInt<TDim, TIdx>,
                        AccCpuThreadsIfAvailableElseInt<TDim, TIdx>,
                        AccCpuFibersIfAvailableElseInt<TDim, TIdx>,
                        AccCpuTbbIfAvailableElseInt<TDim, TIdx>,
                        AccCpuOmp2BlocksIfAvailableElseInt<TDim, TIdx>,
                        AccCpuOmp2ThreadsIfAvailableElseInt<TDim, TIdx>,
                        AccCpuOmp4IfAvailableElseInt<TDim, TIdx>,
                        AccGpuUniformCudaHipRtIfAvailableElseInt<TDim, TIdx>,
                        AccGpuCudaRtIfAvailableElseInt<TDim, TIdx>,
                        AccGpuHipRtIfAvailableElseInt<TDim, TIdx>
                    >;
            }

            //#############################################################################
            //! A vector containing all available accelerators.
            template<
                typename TDim,
                typename TIdx>
            using EnabledAccs =
                typename alpaka::meta::Filter<
                    detail::EnabledAccsElseInt<TDim, TIdx>,
                    std::is_class
                >;

            namespace detail
            {
                //#############################################################################
                //! The accelerator name write wrapper.
                struct StreamOutAccName
                {
                    template<
                        typename TAcc>
                    ALPAKA_FN_HOST auto operator()(
                        std::ostream & os)
                    -> void
                    {
                        os << alpaka::acc::getAccName<TAcc>();
                        os << " ";
                    }
                };
            }

            //-----------------------------------------------------------------------------
            //! Writes the enabled accelerators to the given stream.
            template<
                typename TDim,
                typename TIdx>
            ALPAKA_FN_HOST auto writeEnabledAccs(
                std::ostream & os)
            -> void
            {
                os << "Accelerators enabled: ";

                alpaka::meta::forEachType<
                    EnabledAccs<TDim, TIdx>>(
                        detail::StreamOutAccName(),
                        std::ref(os));

                os << std::endl;
            }

            namespace detail
            {
                //#############################################################################
                //! A std::tuple holding multiple std::tuple consisting of a dimension and a idx type.
                //!
                //! TestDimIdxTuples =
                //!     tuple<
                //!         tuple<Dim1,Idx1>,
                //!         tuple<Dim2,Idx1>,
                //!         tuple<Dim3,Idx1>,
                //!         ...,
                //!         tuple<DimN,IdxN>>
                using TestDimIdxTuples =
                    alpaka::meta::CartesianProduct<
                        std::tuple,
                        dim::TestDims,
                        idx::TestIdxs
                    >;

                //#############################################################################
                //! Transforms a std::tuple holding a dimension and a idx type to a fully instantiated accelerator.
                //!
                //! EnabledAccs<Dim,Idx> = tuple<Acc1<Dim,Idx>, ..., AccN<Dim,Idx>>
                template<
                    typename TTestAccParamSet>
                struct InstantiateEnabledAccsWithTestParamSetImpl
                {
                    using type =
                        EnabledAccs<
                            std::tuple_element_t<0, TTestAccParamSet>,
                            std::tuple_element_t<1, TTestAccParamSet>
                        >;
                };

                template<
                    typename TTestAccParamSet>
                using InstantiateEnabledAccsWithTestParamSet = typename InstantiateEnabledAccsWithTestParamSetImpl<TTestAccParamSet>::type;

                //#############################################################################
                //! A std::tuple containing std::tuple with fully instantiated accelerators.
                //!
                //! TestEnabledAccs =
                //!     tuple<
                //!         tuple<Acc1<Dim1,Idx1>, ..., AccN<Dim1,Idx1>>,
                //!         tuple<Acc1<Dim2,Idx1>, ..., AccN<Dim2,Idx1>>,
                //!         ...,
                //!         tuple<Acc1<DimN,IdxN>, ..., AccN<DimN,IdxN>>>
                using InstantiatedEnabledAccs =
                    alpaka::meta::Transform<
                        TestDimIdxTuples,
                        InstantiateEnabledAccsWithTestParamSet
                    >;
            }

            //#############################################################################
            //! A std::tuple containing fully instantiated accelerators.
            //!
            //! TestAccs =
            //!     tuple<
            //!         Acc1<Dim1,Idx1>, ..., AccN<Dim1,Idx1>,
            //!         Acc1<Dim2,Idx1>, ..., AccN<Dim2,Idx1>,
            //!         ...,
            //!         Acc1<DimN,IdxN>, ..., AccN<DimN,IdxN>>
            using TestAccs =
                alpaka::meta::Apply<
                    detail::InstantiatedEnabledAccs,
                    alpaka::meta::Concatenate
                >;
        }
    }
}
