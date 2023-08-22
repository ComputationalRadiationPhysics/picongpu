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
    class ViewPlainPtr final : public internal::ViewAccessOps<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
    {
        static_assert(!std::is_const_v<TIdx>, "The idx type of the view can not be const!");

        using Dev = alpaka::Dev<TDev>;

    public:
        template<typename TExtent>
        ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, Dev dev, TExtent const& extent = TExtent())
            : m_pMem(pMem)
            , m_dev(std::move(dev))
            , m_extentElements(getExtentVecEnd<TDim>(extent))
            , m_pitchBytes(detail::calculatePitchesFromExtents<TElem>(m_extentElements))
        {
        }

        template<typename TExtent, typename TPitch>
        ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, Dev const dev, TExtent const& extent, TPitch const& pitchBytes)
            : m_pMem(pMem)
            , m_dev(dev)
            , m_extentElements(getExtentVecEnd<TDim>(extent))
            , m_pitchBytes(subVecEnd<TDim>(static_cast<Vec<TDim, TIdx>>(pitchBytes)))
        {
        }

        ViewPlainPtr(ViewPlainPtr const&) = default;
        ALPAKA_FN_HOST
        ViewPlainPtr(ViewPlainPtr&& other) noexcept
            : m_pMem(other.m_pMem)
            , m_dev(other.m_dev)
            , m_extentElements(other.m_extentElements)
            , m_pitchBytes(other.m_pitchBytes)
        {
        }
        ALPAKA_FN_HOST
        auto operator=(ViewPlainPtr const&) -> ViewPlainPtr& = delete;
        ALPAKA_FN_HOST
        auto operator=(ViewPlainPtr&&) -> ViewPlainPtr& = delete;

    public:
        TElem* const m_pMem;
        Dev const m_dev;
        Vec<TDim, TIdx> const m_extentElements;
        Vec<TDim, TIdx> const m_pitchBytes;
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
        //! The ViewPlainPtr width get trait specialization.
        template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetExtent<
            TIdxIntegralConst,
            ViewPlainPtr<TDev, TElem, TDim, TIdx>,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_FN_HOST
            static auto getExtent(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& extent) -> TIdx
            {
                return extent.m_extentElements[TIdxIntegralConst::value];
            }
        };
    } // namespace trait

    namespace trait
    {
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

        //! The ViewPlainPtr memory pitch get trait specialization.
        template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
            struct GetPitchBytes < TIdxIntegralConst,
            ViewPlainPtr<TDev, TElem, TDim, TIdx>, std::enable_if_t<TIdxIntegralConst::value<TDim::value>>
        {
            ALPAKA_FN_HOST static auto getPitchBytes(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> TIdx
            {
                return view.m_pitchBytes[TIdxIntegralConst::value];
            }
        };

        //! The CPU device CreateStaticDevMemView trait specialization.
        template<>
        struct CreateStaticDevMemView<DevCpu>
        {
            template<typename TElem, typename TExtent>
            static auto createStaticDevMemView(TElem* pMem, DevCpu const& dev, TExtent const& extent)
            {
                return alpaka::ViewPlainPtr<DevCpu, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                    pMem,
                    dev,
                    extent);
            }
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        //! The CUDA/HIP RT device CreateStaticDevMemView trait specialization.
        template<typename TApi>
        struct CreateStaticDevMemView<DevUniformCudaHipRt<TApi>>
        {
            template<typename TElem, typename TExtent>
            static auto createStaticDevMemView(
                TElem* pMem,
                DevUniformCudaHipRt<TApi> const& dev,
                TExtent const& extent)
            {
                TElem* pMemAcc(nullptr);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *pMem));

                return alpaka::
                    ViewPlainPtr<DevUniformCudaHipRt<TApi>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                        pMemAcc,
                        dev,
                        extent);
            }
        };
#endif

#if defined(ALPAKA_ACC_SYCL_ENABLED)
        //! The SYCL device CreateStaticDevMemView trait specialization.
        template<typename TPlatform>
        struct CreateStaticDevMemView<DevGenericSycl<TPlatform>>
        {
            static_assert(
                meta::DependentFalseType<TPlatform>::value,
                "The SYCL backend does not support global device variables.");
        };
#endif

        //! The CPU device CreateViewPlainPtr trait specialization.
        template<>
        struct CreateViewPlainPtr<DevCpu>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(DevCpu const& dev, TElem* pMem, TExtent const& extent, TPitch const& pitch)
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
                TPitch const& pitch)
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
                TPitch const& pitch)
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
        template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, ViewPlainPtr<TDev, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST
            static auto getOffset(ViewPlainPtr<TDev, TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
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
