/* Copyright 2024 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci, Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"
#include "alpaka/mem/buf/sycl/ConstBufGenericSycl.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

namespace alpaka::trait
{
    //! The SYCL device memory const-buffer type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct ConstBufType<DevGenericSycl<TTag>, TElem, TDim, TIdx>
    {
        using type = ConstBufGenericSycl<TElem, TDim, TIdx, TTag>;
    };

    //! The ConstBufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DevType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = DevGenericSycl<TTag>;
    };

    //! The ConstBufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getDev(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The ConstBufGenericSycl dimension getter trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct DimType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TDim;
    };

    //! The ConstBufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct ElemType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TElem const;
    };

    //! The ConstBufGenericSycl extent get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetExtents<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf) const
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The ConstBufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrNative<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The ConstBufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>, DevGenericSycl<TTag>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf,
            DevGenericSycl<TTag> const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spBufImpl->m_pMem;
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    //! The ConstBufGenericSycl offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetOffsets<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The ConstBufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct IdxType<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on SYCL device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevGenericSycl<TTag>>
    {
        static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevGenericSycl<TTag> const&) -> TElem const*
        {
            return getPtrNative(buf);
        }
    };

    //! The MakeConstBuf trait for constant Sycl buffers.
    template<typename TElem, typename TDim, typename TIdx, concepts::Tag TTag>
    struct MakeConstBuf<ConstBufGenericSycl<TElem, TDim, TIdx, TTag>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufGenericSycl<TElem, TDim, TIdx, TTag> const& buf)
            -> ConstBufGenericSycl<TElem, TDim, TIdx, TTag>
        {
            return buf;
        }

        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufGenericSycl<TElem, TDim, TIdx, TTag>&& buf)
            -> ConstBufGenericSycl<TElem, TDim, TIdx, TTag>
        {
            return buf;
        }
    };
} // namespace alpaka::trait

#endif
