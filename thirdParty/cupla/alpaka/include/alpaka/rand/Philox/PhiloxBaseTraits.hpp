/* Copyright 2022 Jiří Vyskočil, Bernhard Manfred Gruber, Jeffrey Kelling, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/PhiloxBaseCommon.hpp"
#include "alpaka/rand/Philox/PhiloxBaseCudaArray.hpp"
#include "alpaka/rand/Philox/PhiloxBaseStdArray.hpp"
#include "alpaka/rand/Philox/PhiloxStateless.hpp"
#include "alpaka/rand/Philox/PhiloxStatelessKeyedBase.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
namespace alpaka
{
    template<typename TApi, typename TDim, typename TIdx>
    class AccGpuUniformCudaHipRt;
} // namespace alpaka
#endif

namespace alpaka::rand::engine::trait
{
    template<typename TAcc>
    inline constexpr bool isGPU = false;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    template<typename TApi, typename TDim, typename TIdx>
    inline constexpr bool isGPU<AccGpuUniformCudaHipRt<TApi, TDim, TIdx>> = true;
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
