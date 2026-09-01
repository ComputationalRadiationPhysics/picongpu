/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"
#include "alpaka/mem/buf/sycl/BufGenericSyclImpl.hpp"
#include "alpaka/mem/buf/sycl/ConstBufGenericSycl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka
{
    //! The generic memory buffer template implementing muting accessors.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class BufGenericSycl
        : public internal::ViewAccessorType<DevGenericSycl<TTag>, BufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using TBufImpl = detail::BufGenericSyclImpl<TElem, TDim, TIdx, TTag>;

    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufGenericSycl(
            DevGenericSycl<TTag> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TExtent const& extent)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };
} // namespace alpaka

#    include "alpaka/mem/buf/sycl/Copy.hpp"
#    include "alpaka/mem/buf/sycl/Fill.hpp"
#    include "alpaka/mem/buf/sycl/Set.hpp"
#    include "alpaka/mem/buf/sycl/traits/BufGenericSyclTraits.hpp"
#    include "alpaka/mem/buf/sycl/traits/ConstBufGenericSyclTraits.hpp"

#endif
