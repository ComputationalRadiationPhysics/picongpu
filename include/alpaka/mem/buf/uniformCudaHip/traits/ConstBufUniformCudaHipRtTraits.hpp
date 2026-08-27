/* Copyright 2025 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpu.hpp"
#include "alpaka/mem/buf/uniformCudaHip/BufUniformCudaHipRt.hpp"
#include "alpaka/mem/buf/uniformCudaHip/ConstBufUniformCudaHipRt.hpp"
#include "alpaka/mem/view/Traits.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::trait
{
    //! The CUDA/HIP RT device memory const-buffer type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct ConstBufType<DevUniformCudaHipRt<TApi>, TElem, TDim, TIdx>
    {
        using type = ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>;
    };

    //! The ConstBufUniformCudaHipRt device type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct DevType<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = DevUniformCudaHipRt<TApi>;
    };

    //! The ConstBufUniformCudaHipRt device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getDev(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> DevUniformCudaHipRt<TApi>
        {
            return buf.m_spBufImpl->m_dev;
        }
    };

    //! The ConstBufUniformCudaHipRt dimension getter trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct DimType<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TDim;
    };

    //! The ConstBufUniformCudaHipRt memory element type get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct ElemType<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TElem const;
    };

    //! The ConstBufUniformCudaHipRt extent get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetExtents<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf) const
        {
            return buf.m_spBufImpl->m_extentElements;
        }
    };

    //! The ConstBufUniformCudaHipRt native pointer get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrNative<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> TElem const*
        {
            return buf.m_spBufImpl->m_pMem;
        }
    };

    //! The ConstBufUniformCudaHipRt pointer on device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf,
            DevUniformCudaHipRt<TApi> const& dev) -> TElem const*
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

    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPitchesInBytes<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf) const
            -> Vec<TDim, TIdx>
        {
            if constexpr(TDim::value <= 1)
            {
                return alpaka::detail::calculatePitchesFromExtents<TElem>(buf.m_spBufImpl->m_extentElements);
            }
            else
            {
                return alpaka::detail::calculatePitchesFromExtentsAndPitch<TElem>(
                    buf.m_spBufImpl->m_extentElements,
                    buf.m_spBufImpl->m_rowPitchInBytes);
            }
        }
    };

    //! The ConstBufUniformCudaHipRt offset get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetOffsets<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST auto operator()(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const&) const
            -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The ConstBufUniformCudaHipRt idx type trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct IdxType<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        using type = TIdx;
    };

    //! The ConstBufCpu pointer on CUDA/HIP device get trait specialization.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
    {
        ALPAKA_FN_HOST static auto getPtrDev(
            ConstBufCpu<TElem, TDim, TIdx> const& buf,
            DevUniformCudaHipRt<TApi> const&) -> TElem const*
        {
            // TODO: Check if the memory is mapped at all!
            TElem* pDev(nullptr);

            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostGetDevicePointer(
                &pDev,
                const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                0));

            return pDev;
        }
    };

    //! The MakeConstBuf trait for constant CUDA/HIP buffers.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    struct MakeConstBuf<ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
            -> ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
        {
            return buf;
        }

        ALPAKA_FN_HOST static auto makeConstBuf(ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>&& buf)
            -> ConstBufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
        {
            return buf;
        }
    };

} // namespace alpaka::trait

#endif
