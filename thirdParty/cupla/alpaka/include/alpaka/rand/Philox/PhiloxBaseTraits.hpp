/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseCommon.hpp>
#include <alpaka/rand/Philox/PhiloxBaseStdArray.hpp>
#include <alpaka/rand/Philox/PhiloxStateless.hpp>
#include <alpaka/rand/Philox/PhiloxStatelessKeyedBase.hpp>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/acc/AccGpuUniformCudaHipRt.hpp>
#    include <alpaka/rand/Philox/PhiloxBaseCudaArray.hpp>
#endif

namespace alpaka::rand::engine::trait
{
#if BOOST_COMP_CLANG
    /* TODO: Remove the following pragmas once support for clang 6 is removed. They are necessary because these
    /  clang versions incorrectly warn about a missing 'extern'. */
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#endif
    template<typename TAcc>
    constexpr inline bool isGPU = false;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    template<typename TApi, typename TDim, typename TIdx>
    constexpr inline bool isGPU<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> = true;
#endif
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    /** Selection of default backend
     *
     * Selects the data backend based on the accelerator device type. As of now, different backends operate
     * on different array types.
     *
     * @tparam TAcc the accelerator as defined in alpaka/acc
     * @tparam TParams Philox algorithm parameters
     * @tparam TSfinae internal parameter to stop substitution search and provide the default
     */
    template<typename TAcc, typename TParams, typename TSfinae = void>
    struct PhiloxStatelessBaseTraits
    {
        // template <typename Acc, typename TParams, typename TImpl>
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        using Backend = std::conditional_t<isGPU<TAcc>, PhiloxBaseCudaArray<TParams>, PhiloxBaseStdArray<TParams>>;
#else
        using Backend = PhiloxBaseStdArray<TParams>;
#endif
        using Counter = typename Backend::Counter; ///< Counter array type
        using Key = typename Backend::Key; ///< Key array type
        template<typename TDistributionResultScalar>
        using ResultContainer =
            typename Backend::template ResultContainer<TDistributionResultScalar>; ///< Distribution
                                                                                   ///< container type
        /// Base type to be inherited from by stateless keyed engine
        using Base = PhiloxStateless<Backend, TParams>;
    };

    /** Selection of default backend
     *
     * Selects the data backend based on the accelerator device type. As of now, different backends operate
     * on different array types.
     *
     * @tparam TAcc the accelerator as defined in alpaka/acc
     * @tparam TParams Philox algorithm parameters
     * @tparam TSfinae internal parameter to stop substitution search and provide the default
     */
    template<typename TAcc, typename TParams, typename TSfinae = void>
    struct PhiloxStatelessKeyedBaseTraits : public PhiloxStatelessBaseTraits<TAcc, TParams>
    {
        using Backend = typename PhiloxStatelessBaseTraits<TAcc, TParams>::Backend;
        /// Base type to be inherited from by counting engines
        using Base = PhiloxStatelessKeyedBase<Backend, TParams>;
    };

    /** Selection of default backend
     *
     * Selects the data backend based on the accelerator device type. As of now, different backends operate
     * on different array types.
     *
     * @tparam TAcc the accelerator as defined in alpaka/acc
     * @tparam TParams Philox algorithm parameters
     * @tparam TImpl engine type implementation (CRTP)
     * @tparam TSfinae internal parameter to stop substitution search and provide the default
     */
    template<typename TAcc, typename TParams, typename TImpl, typename TSfinae = void>
    struct PhiloxBaseTraits : public PhiloxStatelessBaseTraits<TAcc, TParams>
    {
        using Backend = typename PhiloxStatelessBaseTraits<TAcc, TParams>::Backend;
        /// Base type to be inherited from by counting engines
        using Base = PhiloxBaseCommon<Backend, TParams, TImpl>;
    };
} // namespace alpaka::rand::engine::trait
