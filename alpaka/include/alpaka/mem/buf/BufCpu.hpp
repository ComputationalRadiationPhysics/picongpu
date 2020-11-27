/* Copyright 2019 Alexander Matthes, Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Unused.hpp>
#include <alpaka/core/Vectorize.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <alpaka/core/Cuda.hpp>
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/core/Hip.hpp>
#endif

#include <alpaka/mem/alloc/AllocCpuAligned.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <memory>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The CPU memory buffer.
        template<typename TElem, typename TDim, typename TIdx>
        class BufCpuImpl final
            : public AllocCpuAligned<std::integral_constant<std::size_t, core::vectorization::defaultAlignment>>
        {
            static_assert(
                !std::is_const<TElem>::value,
                "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
                "elements!");
            static_assert(!std::is_const<TIdx>::value, "The idx type of the buffer can not be const!");

        public:
            //-----------------------------------------------------------------------------
            template<typename TExtent>
            ALPAKA_FN_HOST BufCpuImpl(DevCpu const& dev, TExtent const& extent)
                : AllocCpuAligned<std::integral_constant<std::size_t, core::vectorization::defaultAlignment>>()
                , m_dev(dev)
                , m_extentElements(extent::getExtentVecEnd<TDim>(extent))
                , m_pMem(alpaka::malloc<TElem>(*this, static_cast<std::size_t>(computeElementCount(extent))))
                , m_pitchBytes(static_cast<TIdx>(extent::getWidth(extent) * static_cast<TIdx>(sizeof(TElem))))
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                , m_bPinned(false)
#endif
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    TDim::value == Dim<TExtent>::value,
                    "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                    "identical!");
                static_assert(
                    std::is_same<TIdx, Idx<TExtent>>::value,
                    "The idx type of TExtent and the TIdx template parameter have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem)
                          << " pitch: " << m_pitchBytes << std::endl;
#endif
            }
            //-----------------------------------------------------------------------------
            BufCpuImpl(BufCpuImpl const&) = delete;
            //-----------------------------------------------------------------------------
            BufCpuImpl(BufCpuImpl&&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(BufCpuImpl const&) -> BufCpuImpl& = delete;
            //-----------------------------------------------------------------------------
            auto operator=(BufCpuImpl&&) -> BufCpuImpl& = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~BufCpuImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                // Unpin this memory if it is currently pinned.
                unpin(*this);
#endif
                // NOTE: m_pMem is allowed to be a nullptr here.
                alpaka::free(*this, m_pMem);
            }

        private:
            //-----------------------------------------------------------------------------
            //! \return The number of elements to allocate.
            template<typename TExtent>
            ALPAKA_FN_HOST static auto computeElementCount(TExtent const& extent) -> TIdx
            {
                auto const extentElementCount(extent::getExtentProduct(extent));

                return extentElementCount;
            }

        public:
            DevCpu const m_dev;
            Vec<TDim, TIdx> const m_extentElements;
            TElem* const m_pMem;
            TIdx const m_pitchBytes;
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
            bool m_bPinned;
#endif
        };
    } // namespace detail

    //#############################################################################
    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu
    {
    public:
        //-----------------------------------------------------------------------------
        template<typename TExtent>
        ALPAKA_FN_HOST BufCpu(DevCpu const& dev, TExtent const& extent)
            : m_spBufCpuImpl(std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, extent))
        {
        }
        //-----------------------------------------------------------------------------
        BufCpu(BufCpu const&) = default;
        //-----------------------------------------------------------------------------
        BufCpu(BufCpu&&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(BufCpu const&) -> BufCpu& = default;
        //-----------------------------------------------------------------------------
        auto operator=(BufCpu&&) -> BufCpu& = default;
        //-----------------------------------------------------------------------------
        ~BufCpu() = default;

    public:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;
    };

    namespace traits
    {
        //#############################################################################
        //! The BufCpu device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = DevCpu;
        };
        //#############################################################################
        //! The BufCpu device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(BufCpu<TElem, TDim, TIdx> const& buf) -> DevCpu
            {
                return buf.m_spBufCpuImpl->m_dev;
            }
        };

        //#############################################################################
        //! The BufCpu dimension getter trait.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The BufCpu memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };
    } // namespace traits
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCpu width get trait specialization.
            template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                BufCpu<TElem, TDim, TIdx>,
                std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(BufCpu<TElem, TDim, TIdx> const& extent) -> TIdx
                {
                    return extent.m_spBufCpuImpl->m_extentElements[TIdxIntegralConst::value];
                }
            };
        } // namespace traits
    } // namespace extent
    namespace traits
    {
        //#############################################################################
        //! The BufCpu native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
        };
        //#############################################################################
        //! The BufCpu pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevCpu const& dev)
                -> TElem const*
            {
                if(dev == getDev(buf))
                {
                    return buf.m_spBufCpuImpl->m_pMem;
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> TElem*
            {
                if(dev == getDev(buf))
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
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPitchBytes<DimInt<TDim::value - 1u>, BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getPitchBytes(BufCpu<TElem, TDim, TIdx> const& pitch) -> TIdx
            {
                return pitch.m_spBufCpuImpl->m_pitchBytes;
            }
        };

        //#############################################################################
        //! The BufCpu memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAlloc<TElem, TDim, TIdx, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevCpu const& dev, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                return BufCpu<TElem, TDim, TIdx>(dev, extent);
            }
        };
        //#############################################################################
        //! The BufCpu memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Map<BufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto map(BufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    throw std::runtime_error("Memory mapping of BufCpu between two devices is not implemented!");
                }
                // If it is the same device, nothing has to be mapped.
            }
        };
        //#############################################################################
        //! The BufCpu memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unmap<BufCpu<TElem, TDim, TIdx>, DevCpu>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unmap(BufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(getDev(buf) != dev)
                {
                    throw std::runtime_error("Memory unmapping of BufCpu between two devices is not implemented!");
                }
                // If it is the same device, nothing has to be mapped.
            }
        };
        //#############################################################################
        //! The BufCpu memory pinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Pin<BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto pin(BufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(!isPinned(buf))
                {
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                    if(buf.m_spBufCpuImpl->m_extentElements.prod() != 0)
                    {
                        // - cudaHostRegisterDefault:
                        //   See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                        // - cudaHostRegisterPortable:
                        //   The memory returned by this call will be considered as pinned memory by all CUDA contexts,
                        //   not just the one that performed the allocation.
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                            ALPAKA_API_PREFIX(HostRegister)(
                                const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                                extent::getExtentProduct(buf) * sizeof(Elem<BufCpu<TElem, TDim, TIdx>>),
                                ALPAKA_API_PREFIX(HostRegisterDefault)),
                            ALPAKA_API_PREFIX(ErrorHostMemoryAlreadyRegistered));

                        buf.m_spBufCpuImpl->m_bPinned = true;
                    }
#else
                    static_assert(
                        meta::DependentFalseType<TElem>::value,
                        "Memory pinning of BufCpu is not implemented when CUDA or HIP is not enabled!");
#endif
                }
            }
        };
        //#############################################################################
        //! The BufCpu memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unpin(BufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                alpaka::unpin(*buf.m_spBufCpuImpl.get());
            }
        };
        //#############################################################################
        //! The BufCpuImpl memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto unpin(alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>& bufImpl) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(isPinned(bufImpl))
                {
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                        ALPAKA_API_PREFIX(HostUnregister)(
                            const_cast<void*>(reinterpret_cast<void const*>(bufImpl.m_pMem))),
                        ALPAKA_API_PREFIX(ErrorHostMemoryNotRegistered));

                    bufImpl.m_bPinned = false;
#else
                    static_assert(
                        meta::DependentFalseType<TElem>::value,
                        "Memory unpinning of BufCpu is not implemented when CUDA or HIP is not enabled!");
#endif
                }
            }
        };
        //#############################################################################
        //! The BufCpu memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isPinned(BufCpu<TElem, TDim, TIdx> const& buf) -> bool
            {
                return alpaka::isPinned(*buf.m_spBufCpuImpl.get());
            }
        };
        //#############################################################################
        //! The BufCpuImpl memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isPinned(alpaka::detail::BufCpuImpl<TElem, TDim, TIdx> const& bufImpl) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                return bufImpl.m_bPinned;
#else
                alpaka::ignore_unused(bufImpl);
                return false;
#endif
            }
        };
        //#############################################################################
        //! The BufCpu memory prepareForAsyncCopy trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct PrepareForAsyncCopy<BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto prepareForAsyncCopy(BufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // to optimize the data transfer performance between a cuda/hip device the cpu buffer has to be pinned,
                // for exclusive cpu use, no preparing is needed
#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA) || (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && BOOST_LANG_HIP)
                pin(buf);
#else
                alpaka::ignore_unused(buf);
#endif
            }
        };

        //#############################################################################
        //! The BufCpu offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, BufCpu<TElem, TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getOffset(BufCpu<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //#############################################################################
        //! The BufCpu idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace traits
} // namespace alpaka

#include <alpaka/mem/buf/cpu/Copy.hpp>
#include <alpaka/mem/buf/cpu/Set.hpp>
