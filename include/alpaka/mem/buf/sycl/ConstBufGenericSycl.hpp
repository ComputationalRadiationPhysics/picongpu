/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/sycl/BufGenericSyclImpl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <memory>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    // Predeclaration
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class BufGenericSycl;

    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    class ConstBufGenericSycl
        : public internal::ViewAccessorType<DevGenericSycl<TTag>, ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
    public:
        //! Constructor
        template<typename TExtent, typename Deleter>
        ConstBufGenericSycl(DevGenericSycl<TTag> const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufImpl{std::make_shared<detail::BufGenericSyclImpl<TElem, TDim, TIdx, TTag>>(
                dev,
                pMem,
                std::move(deleter),
                extent)}
        {
        }

        //! Constructor for a ConstBuf from a BufGenericSycl
        ALPAKA_FN_HOST ConstBufGenericSycl(BufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            : m_spBufImpl{buf.m_spBufImpl}
        {
        }

        //! Constructor for a ConstBuf from a BufGenericSycl
        ALPAKA_FN_HOST ConstBufGenericSycl(BufGenericSycl<TElem, TDim, TIdx, TTag>&& buf)
            : m_spBufImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        std::shared_ptr<detail::BufGenericSyclImpl<TElem, TDim, TIdx, TTag>> m_spBufImpl;

        friend alpaka::trait::GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>;
        friend alpaka::trait::GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>;
    };
} // namespace alpaka

#endif
