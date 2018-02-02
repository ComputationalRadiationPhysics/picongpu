/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevCudaRt.hpp>

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            //#############################################################################
            //! The memory view to wrap plain pointers.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            class ViewPlainPtr final
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent>
                ALPAKA_FN_HOST_ACC ViewPlainPtr(
                    TElem * pMem,
                    TDev const & dev,
                    TExtent const & extent = TExtent()) :
                        m_pMem(pMem),
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_pitchBytes(calculatePitchesFromExtents(m_extentElements))
                {}

                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent,
                    typename TPitch>
                ALPAKA_FN_HOST_ACC ViewPlainPtr(
                    TElem * pMem,
                    TDev const dev,
                    TExtent const & extent,
                    TPitch const & pitchBytes) :
                        m_pMem(pMem),
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_pitchBytes(
                            vec::subVecEnd<TDim>(
                               static_cast<
                                    vec::Vec<TDim, TSize> >(pitchBytes)
                            )
                        )
                {}

                //-----------------------------------------------------------------------------
                ViewPlainPtr(ViewPlainPtr const &) = delete;
                //-----------------------------------------------------------------------------
                ViewPlainPtr(ViewPlainPtr &&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(ViewPlainPtr const &) -> ViewPlainPtr & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(ViewPlainPtr &&) -> ViewPlainPtr & = default;
                //-----------------------------------------------------------------------------
                ~ViewPlainPtr() = default;

            private:
                //-----------------------------------------------------------------------------
                //! Calculate the pitches purely from the extents.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtent>
                ALPAKA_FN_HOST_ACC static auto calculatePitchesFromExtents(
                    TExtent const & extent)
                -> vec::Vec<TDim, TSize>
                {
                    vec::Vec<TDim, TSize> pitchBytes(vec::Vec<TDim, TSize>::all(0));
                    pitchBytes[TDim::value - 1u] = extent[TDim::value - 1u] * static_cast<TSize>(sizeof(TElem));
                    for(TSize i = TDim::value - 1u; i > static_cast<TSize>(0u); --i)
                    {
                        pitchBytes[i-1] = extent[i-1] * pitchBytes[i];
                    }
                    return pitchBytes;
                }

            public:
                TElem * const m_pMem;
                TDev const m_dev;
                vec::Vec<TDim, TSize> const m_extentElements;
                vec::Vec<TDim, TSize> const m_pitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for ViewPlainPtr.
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr device type trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct DevType<
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The ViewPlainPtr device get trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetDev<
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getDev(
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> const & view)
                    -> TDev
                {
                    return view.m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr dimension getter trait.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct DimType<
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr memory element type get trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct ElemType<
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr width get trait specialization.
            template<
                typename TIdx,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> const & extent)
                -> TSize
                {
                    return extent.m_extentElements[TIdx::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The ViewPlainPtr native pointer get trait specialization.
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrNative<
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> const & view)
                    -> TElem const *
                    {
                        return view.m_pMem;
                    }
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> & view)
                    -> TElem *
                    {
                        return view.m_pMem;
                    }
                };

                //#############################################################################
                //! The ViewPlainPtr memory pitch get trait specialization.
                template<
                    typename TIdx,
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    TIdx,
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>,
                    typename std::enable_if<TIdx::value < TDim::value>::type>
                {
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> const & view)
                    -> TSize
                    {
                        return view.m_pitchBytes[TIdx::value];
                    }
                };

                //#############################################################################
                //! The CPU device CreateStaticDevMemView trait specialization.
                template<>
                struct CreateStaticDevMemView<
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem,
                        typename TExtent>
                    ALPAKA_FN_HOST static auto createStaticDevMemView(
                        TElem * pMem,
                        dev::DevCpu const & dev,
                        TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> alpaka::mem::view::ViewPlainPtr<dev::DevCpu, TElem, alpaka::dim::Dim<TExtent>, alpaka::size::Size<TExtent>>
#endif
                    {
                        return
                            alpaka::mem::view::ViewPlainPtr<
                                dev::DevCpu,
                                TElem,
                                alpaka::dim::Dim<TExtent>,
                                alpaka::size::Size<TExtent>>(
                                    pMem,
                                    dev,
                                    extent);
                    }
                };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                //#############################################################################
                //! The CUDA RT device CreateStaticDevMemView trait specialization.
                template<>
                struct CreateStaticDevMemView<
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem,
                        typename TExtent>
                    ALPAKA_FN_HOST static auto createStaticDevMemView(
                        TElem * pMem,
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                    -> alpaka::mem::view::ViewPlainPtr<dev::DevCudaRt, TElem, alpaka::dim::Dim<TExtent>, alpaka::size::Size<TExtent>>
#endif
                    {
                        TElem* pMemAcc(nullptr);
                        ALPAKA_CUDA_RT_CHECK(
                            cudaGetSymbolAddress(
                                reinterpret_cast<void **>(&pMemAcc),
                                *pMem));
                        return
                            alpaka::mem::view::ViewPlainPtr<
                                dev::DevCudaRt,
                                TElem,
                                alpaka::dim::Dim<TExtent>,
                                alpaka::size::Size<TExtent>>(
                                    pMemAcc,
                                    dev,
                                    extent);
                    }
                };
#endif
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr offset get trait specialization.
            template<
                typename TIdx,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize> const &)
                -> TSize
                {
                    return 0u;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The ViewPlainPtr size type trait specialization.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct SizeType<
                mem::view::ViewPlainPtr<TDev, TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}
