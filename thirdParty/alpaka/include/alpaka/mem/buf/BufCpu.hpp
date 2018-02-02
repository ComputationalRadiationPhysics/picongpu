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

#include <alpaka/core/Vectorize.hpp>
#include <alpaka/dev/DevCpu.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <alpaka/vec/Vec.hpp>

// \TODO: Remove CUDA inclusion for BufCpu by replacing pinning with non CUDA code!
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
    #include <alpaka/core/Cuda.hpp>
#endif

#include <alpaka/mem/alloc/AllocCpuBoostAligned.hpp>

#include <alpaka/meta/DependentFalseType.hpp>

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

#include <memory>

namespace alpaka
{
    namespace mem
    {
        namespace buf
        {
            namespace cpu
            {
                namespace detail
                {
                    //#############################################################################
                    //! The CPU memory buffer.
                    template<
                        typename TElem,
                        typename TDim,
                        typename TSize>
                    class BufCpuImpl final :
                        public mem::alloc::AllocCpuBoostAligned<std::integral_constant<std::size_t, core::vectorization::defaultAlignment>>
                    {
                    public:
                        //-----------------------------------------------------------------------------
                        template<
                            typename TExtent>
                        ALPAKA_FN_HOST BufCpuImpl(
                            dev::DevCpu const & dev,
                            TExtent const & extent) :
                                mem::alloc::AllocCpuBoostAligned<std::integral_constant<std::size_t, core::vectorization::defaultAlignment>>(),
                                m_dev(dev),
                                m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                                m_pMem(mem::alloc::alloc<TElem>(*this, static_cast<std::size_t>(computeElementCount(extent)))),
                                m_pitchBytes(static_cast<TSize>(extent::getWidth(extent) * static_cast<TSize>(sizeof(TElem))))
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                                ,m_bPinned(false)
#endif
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                            static_assert(
                                TDim::value == dim::Dim<TExtent>::value,
                                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");
                            static_assert(
                                std::is_same<TSize, size::Size<TExtent>>::value,
                                "The size type of TExtent and the TSize template parameter have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            std::cout << BOOST_CURRENT_FUNCTION
                                << " e: " << m_extentElements
                                << " ptr: " << static_cast<void *>(m_pMem)
                                << " pitch: " << m_pitchBytes
                                << std::endl;
#endif
                        }
                        //-----------------------------------------------------------------------------
                        BufCpuImpl(BufCpuImpl const &) = delete;
                        //-----------------------------------------------------------------------------
                        BufCpuImpl(BufCpuImpl &&) = default;
                        //-----------------------------------------------------------------------------
                        auto operator=(BufCpuImpl const &) -> BufCpuImpl & = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BufCpuImpl &&) -> BufCpuImpl & = default;
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST ~BufCpuImpl() noexcept(false)
                        {
                            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                            // Unpin this memory if it is currently pinned.
                            mem::buf::unpin(*this);
#endif
                            // NOTE: m_pMem is allowed to be a nullptr here.
                            mem::alloc::free(*this, m_pMem);
                        }

                    private:
                        //-----------------------------------------------------------------------------
                        //! \return The number of elements to allocate.
                        template<
                            typename TExtent>
                        ALPAKA_FN_HOST static auto computeElementCount(
                            TExtent const & extent)
                        -> TSize
                        {
                            auto const extentElementCount(extent::getProductOfExtent(extent));

                            return extentElementCount;
                        }

                    public:
                        dev::DevCpu const m_dev;
                        vec::Vec<TDim, TSize> const m_extentElements;
                        TElem * const m_pMem;
                        TSize const m_pitchBytes;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                        bool m_bPinned;
#endif
                    };
                }
            }
            //#############################################################################
            //! The CPU memory buffer.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            class BufCpu
            {
            public:
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                ALPAKA_FN_HOST BufCpu(
                    dev::DevCpu const & dev,
                    TExtent const & extent) :
                        m_spBufCpuImpl(std::make_shared<cpu::detail::BufCpuImpl<TElem, TDim, TSize>>(dev, extent))
                {}
                //-----------------------------------------------------------------------------
                BufCpu(BufCpu const &) = default;
                //-----------------------------------------------------------------------------
                BufCpu(BufCpu &&) = default;
                //-----------------------------------------------------------------------------
                auto operator=(BufCpu const &) -> BufCpu & = default;
                //-----------------------------------------------------------------------------
                auto operator=(BufCpu &&) -> BufCpu & = default;
                //-----------------------------------------------------------------------------
                ~BufCpu() = default;

            public:
                std::shared_ptr<cpu::detail::BufCpuImpl<TElem, TDim, TSize>> m_spBufCpuImpl;
            };
        }
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCpu device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct DevType<
                mem::buf::BufCpu<TElem, TDim, TSize>>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The BufCpu device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetDev<
                mem::buf::BufCpu<TElem, TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufCpu<TElem, TDim, TSize> const & buf)
                -> dev::DevCpu
                {
                    return buf.m_spBufCpuImpl->m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCpu dimension getter trait.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct DimType<
                mem::buf::BufCpu<TElem, TDim, TSize>>
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
            //! The BufCpu memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct ElemType<
                mem::buf::BufCpu<TElem, TDim, TSize>>
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
            //! The BufCpu width get trait specialization.
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::buf::BufCpu<TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufCpu<TElem, TDim, TSize> const & extent)
                -> TSize
                {
                    return extent.m_spBufCpuImpl->m_extentElements[TIdx::value];
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
                //! The BufCpu native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrNative<
                    mem::buf::BufCpu<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufCpu<TElem, TDim, TSize> const & buf)
                    -> TElem const *
                    {
                        return buf.m_spBufCpuImpl->m_pMem;
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf)
                    -> TElem *
                    {
                        return buf.m_spBufCpuImpl->m_pMem;
                    }
                };
                //#############################################################################
                //! The BufCpu pointer on device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TSize> const & buf,
                        dev::DevCpu const & dev)
                    -> TElem const *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return buf.m_spBufCpuImpl->m_pMem;
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
                        dev::DevCpu const & dev)
                    -> TElem *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return buf.m_spBufCpuImpl->m_pMem;
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                };
                //#############################################################################
                //! The BufCpu pitch get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufCpu<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufCpu<TElem, TDim, TSize> const & pitch)
                    -> TSize
                    {
                        return pitch.m_spBufCpuImpl->m_pitchBytes;
                    }
                };
            }
        }
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Alloc<
                    TElem,
                    TDim,
                    TSize,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCpu const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCpu<TElem, TDim, TSize>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        return mem::buf::BufCpu<
                            TElem,
                            TDim,
                            TSize>(
                                dev,
                                extent);
                    }
                };
                //#############################################################################
                //! The BufCpu memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
                        dev::DevCpu const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Memory mapping of BufCpu between two devices is not implemented!");
                        }
                        // If it is the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCpu memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
                        dev::DevCpu const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Memory unmapping of BufCpu between two devices is not implemented!");
                        }
                        // If it is the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCpu memory pinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Pin<
                    mem::buf::BufCpu<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(!mem::buf::isPinned(buf))
                        {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                            if(buf.m_spBufCpuImpl->m_extentElements.prod() != 0)
                            {
                                // - cudaHostRegisterDefault:
                                //   See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                                // - cudaHostRegisterPortable:
                                //   The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
                                ALPAKA_CUDA_RT_CHECK_IGNORE(
                                    cudaHostRegister(
                                        const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                        extent::getProductOfExtent(buf) * sizeof(elem::Elem<buf::BufCpu<TElem, TDim, TSize>>),
                                        cudaHostRegisterDefault),
                                    cudaErrorHostMemoryAlreadyRegistered);

                                buf.m_spBufCpuImpl->m_bPinned = true;
                            }
#else
                            static_assert(
                                meta::DependentFalseType<TElem>::value,
                                "Memory pinning of BufCpu is not implemented when CUDA is not enabled!");
#endif
                        }
                    }
                };
                //#############################################################################
                //! The BufCpu memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Unpin<
                    mem::buf::BufCpu<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf)
                    -> void
                    {
                        mem::buf::unpin(*buf.m_spBufCpuImpl.get());
                    }
                };
                //#############################################################################
                //! The BufCpuImpl memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Unpin<
                    mem::buf::cpu::detail::BufCpuImpl<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::cpu::detail::BufCpuImpl<TElem, TDim, TSize> & bufImpl)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(mem::buf::isPinned(bufImpl))
                        {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                            ALPAKA_CUDA_RT_CHECK_IGNORE(
                                cudaHostUnregister(
                                    const_cast<void *>(reinterpret_cast<void const *>(bufImpl.m_pMem))),
                                cudaErrorHostMemoryNotRegistered);

                            bufImpl.m_bPinned = false;
#else
                            static_assert(
                                meta::DependentFalseType<TElem>::value,
                                "Memory unpinning of BufCpu is not implemented when CUDA is not enabled!");
#endif
                        }
                    }
                };
                //#############################################################################
                //! The BufCpu memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct IsPinned<
                    mem::buf::BufCpu<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufCpu<TElem, TDim, TSize> const & buf)
                    -> bool
                    {
                        return mem::buf::isPinned(*buf.m_spBufCpuImpl.get());
                    }
                };
                //#############################################################################
                //! The BufCpuImpl memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct IsPinned<
                    mem::buf::cpu::detail::BufCpuImpl<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::cpu::detail::BufCpuImpl<TElem, TDim, TSize> const & bufImpl)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
                        return bufImpl.m_bPinned;
#else
                        boost::ignore_unused(bufImpl);
                        return false;
#endif
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCpu offset get trait specialization.
            template<
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::buf::BufCpu<TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    mem::buf::BufCpu<TElem, TDim, TSize> const &)
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
            //! The BufCpu size type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct SizeType<
                mem::buf::BufCpu<TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

#include <alpaka/mem/buf/cpu/Copy.hpp>
#include <alpaka/mem/buf/cpu/Set.hpp>
