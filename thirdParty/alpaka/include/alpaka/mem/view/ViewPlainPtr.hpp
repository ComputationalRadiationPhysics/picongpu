/* Copyright 2023 Benjamin Worpitz, Matthias Werner, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber,
 *                Jan Stephan, Andrea Bocci, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/vec/Vec.hpp"

#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The memory view to wrap plain pointers.
    template<typename TDev, typename TElem, typename TDim, typename TIdx>
    struct ViewPlainPtr final : internal::ViewAccessOps<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
    {
        static_assert(!std::is_const_v<TIdx>, "The idx type of the view can not be const!");

        template<typename TExtent>
        ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, TDev dev, TExtent const& extent = TExtent())
            : ViewPlainPtr(pMem, std::move(dev), extent, detail::calculatePitchesFromExtents<TElem>(extent))
        {
        }

        template<typename TExtent, typename TPitch>
        ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, TDev dev, TExtent const& extent, TPitch pitchBytes)
            : m_pMem(pMem)
            , m_dev(std::move(dev))
            , m_extentElements(extent)
            , m_pitchBytes(static_cast<Vec<TDim, TIdx>>(pitchBytes))
        {
        }

        TElem* m_pMem;
        TDev m_dev;
        Vec<TDim, TIdx> m_extentElements;
        Vec<TDim, TIdx> m_pitchBytes;
    };

    // Trait specializations for ViewPlainPtr.
    namespace trait
    {
        //! The ViewPlainPtr device type trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct DevType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = alpaka::Dev<TDev>;
        };

        //! The ViewPlainPtr device get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetDev<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            static auto getDev(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> alpaka::Dev<TDev>
            {
                return view.m_dev;
            }
        };

        //! The ViewPlainPtr dimension getter trait.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct DimType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The ViewPlainPtr memory element type get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct ElemType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = TElem;
        };
    } // namespace trait

    namespace trait
    {
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetExtents<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) const
            {
                return view.m_extentElements;
            }
        };

        //! The ViewPlainPtr native pointer get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            static auto getPtrNative(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> TElem const*
            {
                return view.m_pMem;
            }

            static auto getPtrNative(ViewPlainPtr<TDev, TElem, TDim, TIdx>& view) -> TElem*
            {
                return view.m_pMem;
            }
        };

        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetPitchesInBytes<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) const
            {
                return view.m_pitchBytes;
            }
        };

        //! The CPU device CreateViewPlainPtr trait specialization.
        template<>
        struct CreateViewPlainPtr<DevCpu>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(DevCpu const& dev, TElem* pMem, TExtent const& extent, TPitch pitch)
            {
                return alpaka::ViewPlainPtr<DevCpu, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                    pMem,
                    dev,
                    extent,
                    pitch);
            }
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        //! The CUDA/HIP RT device CreateViewPlainPtr trait specialization.
        template<typename TApi>
        struct CreateViewPlainPtr<DevUniformCudaHipRt<TApi>>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(
                DevUniformCudaHipRt<TApi> const& dev,
                TElem* pMem,
                TExtent const& extent,
                TPitch pitch)
            {
                return alpaka::
                    ViewPlainPtr<DevUniformCudaHipRt<TApi>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                        pMem,
                        dev,
                        extent,
                        pitch);
            }
        };
#endif

#if defined(ALPAKA_ACC_SYCL_ENABLED)
        //! The SYCL device CreateViewPlainPtr trait specialization.
        template<typename TPlatform>
        struct CreateViewPlainPtr<DevGenericSycl<TPlatform>>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(
                DevGenericSycl<TPlatform> const& dev,
                TElem* pMem,
                TExtent const& extent,
                TPitch pitch)
            {
                return alpaka::
                    ViewPlainPtr<DevGenericSycl<TPlatform>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                        pMem,
                        dev,
                        extent,
                        pitch);
            }
        };
#endif
        //! The ViewPlainPtr offset get trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetOffsets<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST auto operator()(ViewPlainPtr<TDev, TElem, TDim, TIdx> const&) const -> Vec<TDim, TIdx>
            {
                return Vec<TDim, TIdx>::zeros();
            }
        };

        //! The ViewPlainPtr idx type trait specialization.
        template<typename TDev, typename TElem, typename TDim, typename TIdx>
        struct IdxType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka
