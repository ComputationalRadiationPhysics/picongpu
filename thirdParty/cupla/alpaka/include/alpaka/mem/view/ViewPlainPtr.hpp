/* Copyright 2022 Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevOacc.hpp>
#include <alpaka/dev/DevOmp5.hpp>
#include <alpaka/dev/DevUniformCudaHipRt.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewAccessOps.hpp>
#include <alpaka/vec/Vec.hpp>

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

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        //! The Omp5 device CreateStaticDevMemView trait specialization.
        template<>
        struct CreateStaticDevMemView<DevOmp5>
        {
            template<typename TElem, typename TExtent>
            static auto createStaticDevMemView(TElem* pMem, DevOmp5 const& dev, TExtent const& extent)
            {
                return alpaka::ViewPlainPtr<DevOmp5, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                    dev.mapStatic(pMem, extent),
                    dev,
                    extent);
            }
        };
#endif

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
        //! The Oacc device CreateStaticDevMemView trait specialization.
        template<>
        struct CreateStaticDevMemView<DevOacc>
        {
            template<typename TElem, typename TExtent>
            static auto createStaticDevMemView(TElem* pMem, DevOacc const& dev, TExtent const& extent)
            {
                return alpaka::ViewPlainPtr<DevOacc, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                    dev.mapStatic(pMem, extent),
                    dev,
                    extent);
            }
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

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        //! The Omp5 device CreateViewPlainPtr trait specialization.
        //! \todo What ist this for? Does this exist in OMP5?
        template<>
        struct CreateViewPlainPtr<DevOmp5>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(DevOmp5 const& dev, TElem* pMem, TExtent const& extent, TPitch const& pitch)
            {
                return alpaka::ViewPlainPtr<DevOmp5, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
                    pMem,
                    dev,
                    extent,
                    pitch);
            }
        };
#endif

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED
        //! The Oacc device CreateViewPlainPtr trait specialization.
        template<>
        struct CreateViewPlainPtr<DevOacc>
        {
            template<typename TElem, typename TExtent, typename TPitch>
            static auto createViewPlainPtr(DevOacc const& dev, TElem* pMem, TExtent const& extent, TPitch const& pitch)
            {
                return alpaka::ViewPlainPtr<DevOacc, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
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
