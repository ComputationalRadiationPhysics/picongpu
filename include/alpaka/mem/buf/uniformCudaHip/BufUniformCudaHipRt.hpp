/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include "alpaka/dev/Traits.hpp"
#    include "alpaka/mem/buf/Traits.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRt.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRtImpl.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/ConstBufUniformCudaHipRt.hpp"
#    include "alpaka/mem/view/ViewAccessOps.hpp"
#    include "alpaka/vec/Vec.hpp"

#    include <functional>
#    include <memory>
#    include <type_traits>
#    include <utility>

namespace alpaka
{

    //! The generic memory buffer template implementing muting accessors.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt
        : public internal::ViewAccessorType<DevUniformCudaHipRt<TApi>, BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using TBufImpl = detail::BufUniformCudaHipRtImpl<TApi, TElem, TDim, TIdx>;

    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufUniformCudaHipRt(
            DevUniformCudaHipRt<TApi> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent,
            std::size_t pitchBytes)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent, pitchBytes)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };

} // namespace alpaka

#    include "alpaka/mem/buf/uniformCudaHip/Copy.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/Fill.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/Set.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/traits/BufUniformCudaHipRtTraits.hpp"
#    include "alpaka/mem/buf/uniformCudaHip/traits/ConstBufUniformCudaHipRtTraits.hpp"

#endif
