/* Copyright 2025 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRtImpl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/vec/Vec.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    // Forward declarations.
    struct ApiCudaRt;
    struct ApiHipRt;
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt;

    //! The CUDA/HIP memory buffer.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct ConstBufUniformCudaHipRt
        : internal::ViewAccessorType<DevUniformCudaHipRt<TApi>, ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        static_assert(!std::is_const_v<TElem>, "The elem type of the buffer must not be const");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer must not be const!");

        //! Constructor
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST ConstBufUniformCudaHipRt(
            DevUniformCudaHipRt<TApi> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent,
            std::size_t pitchBytes)
            : m_spBufImpl{std::make_shared<detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>>(
                dev,
                pMem,
                std::move(deleter),
                extent,
                pitchBytes)}
        {
        }

        ALPAKA_FN_HOST ConstBufUniformCudaHipRt(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            : m_spBufImpl{buf.m_spBufImpl}
        {
        }

        ALPAKA_FN_HOST ConstBufUniformCudaHipRt(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>&& buf)
            : m_spBufImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        std::shared_ptr<detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>> m_spBufImpl;

        friend alpaka::trait::GetDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
        friend alpaka::trait::GetExtents<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrNative<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>;
        friend alpaka::trait::GetPitchesInBytes<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>;
    };

} // namespace alpaka

#    include "alpaka/mem/buf/uniformCudaHip/Copy.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/Fill.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/Set.hpp"

#endif
