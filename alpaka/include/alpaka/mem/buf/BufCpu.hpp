/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Vectorize.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/ViewAccessOps.hpp>
#include <alpaka/vec/Vec.hpp>

// Backend specific includes.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/core/Cuda.hpp>
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/core/Hip.hpp>
#endif

#include <alpaka/mem/alloc/AllocCpuAligned.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace detail
    {
        //! The CPU memory buffer.
        template<typename TElem, typename TDim, typename TIdx>
        class BufCpuImpl final
        {
            static_assert(
                !std::is_const_v<TElem>,
                "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
                "elements!");
            static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        public:
            template<typename TExtent>
            ALPAKA_FN_HOST BufCpuImpl(
                DevCpu dev,
                TElem* pMem,
                std::function<void(TElem*)> deleter,
                TExtent const& extent) noexcept
                : m_dev(std::move(dev))
                , m_extentElements(getExtentVecEnd<TDim>(extent))
                , m_pMem(pMem)
                , m_deleter(std::move(deleter))
                , m_bPinned(false)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    TDim::value == Dim<TExtent>::value,
                    "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                    "identical!");
                static_assert(
                    std::is_same_v<TIdx, Idx<TExtent>>,
                    "The idx type of TExtent and the TIdx template parameter have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem)
                          << std::endl;
#endif
            }
            BufCpuImpl(BufCpuImpl&&) = delete;
            auto operator=(BufCpuImpl&&) -> BufCpuImpl& = delete;
            ALPAKA_FN_HOST ~BufCpuImpl()
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Unpin this memory if it is currently pinned.
                if(m_bPinned)
                {
                    try
                    {
                        unpin(*this); // May throw std::runtime_error
                    }
                    catch(std::runtime_error const& err)
                    {
                        std::cerr << "Caught runtime error while unpinning in ~BufCpuImpl(): " << err.what()
                                  << std::endl;
                    }
                }
                // NOTE: m_pMem is allowed to be a nullptr here.
                m_deleter(m_pMem);
            }

        public:
            DevCpu const m_dev;
            Vec<TDim, TIdx> const m_extentElements;
            TElem* const m_pMem;
            std::function<void(TElem*)> m_deleter;
            bool m_bPinned;
        };
    } // namespace detail

    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu : public internal::ViewAccessOps<BufCpu<TElem, TDim, TIdx>>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufCpu(DevCpu const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufCpuImpl{
                std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
        }

    public:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;
    };

    namespace trait
    {
        //! The BufCpu device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = DevCpu;
        };
        //! The BufCpu device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(BufCpu<TElem, TDim, TIdx> const& buf) -> DevCpu
            {
                return buf.m_spBufCpuImpl->m_dev;
            }
        };

        //! The BufCpu dimension getter trait.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The BufCpu memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        //! The BufCpu width get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetExtent<
            TIdxIntegralConst,
            BufCpu<TElem, TDim, TIdx>,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_FN_HOST static auto getExtent(BufCpu<TElem, TDim, TIdx> const& extent) -> TIdx
            {
                return extent.m_spBufCpuImpl->m_extentElements[TIdxIntegralConst::value];
            }
        };

        //! The BufCpu native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
            ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_spBufCpuImpl->m_pMem;
            }
        };
        //! The BufCpu pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevCpu>
        {
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

        //! The BufCpu memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAlloc<TElem, TDim, TIdx, DevCpu>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevCpu const& dev, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // If ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT is defined, positive, and a power of 2, use it as the
                // default alignment for host memory allocations. Otherwise, the alignment is chosen to enable optimal
                // performance dependant on the target architecture.
#if defined(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT)
                static_assert(
                    ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT > 0
                        && ((ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT & (ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT - 1)) == 0),
                    "If defined, ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT must be a power of 2.");
                constexpr std::size_t alignment = static_cast<std::size_t>(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT);
#else
                constexpr std::size_t alignment = core::vectorization::defaultAlignment;
#endif
                // alpaka::AllocCpuAligned is stateless
                using Allocator = AllocCpuAligned<std::integral_constant<std::size_t, alignment>>;
                static_assert(std::is_empty_v<Allocator>, "AllocCpuAligned is expected to be stateless");
                auto* memPtr = alpaka::malloc<TElem>(Allocator{}, static_cast<std::size_t>(getExtentProduct(extent)));
                auto deleter = [](TElem* ptr) { alpaka::free(Allocator{}, ptr); };

                return BufCpu<TElem, TDim, TIdx>(dev, memPtr, std::move(deleter), extent);
            }
        };
        //! The BufCpu stream-ordered memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct AsyncBufAlloc<TElem, TDim, TIdx, DevCpu>
        {
            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    std::is_same_v<Dev<TQueue>, DevCpu>,
                    "The BufCpu buffer can only be used with a queue on a DevCpu device!");
                DevCpu const& dev = getDev(queue);

                // If ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT is defined, positive, and a power of 2, use it as the
                // default alignment for host memory allocations. Otherwise, the alignment is chosen to enable optimal
                // performance dependant on the target architecture.
#if defined(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT)
                static_assert(
                    ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT > 0
                        && ((ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT & (ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT - 1)) == 0),
                    "If defined, ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT must be a power of 2.");
                constexpr std::size_t alignment = static_cast<std::size_t>(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT);
#else
                constexpr std::size_t alignment = core::vectorization::defaultAlignment;
#endif
                // alpaka::AllocCpuAligned is stateless
                using Allocator = AllocCpuAligned<std::integral_constant<std::size_t, alignment>>;
                static_assert(std::is_empty_v<Allocator>, "AllocCpuAligned is expected to be stateless");
                auto* memPtr = alpaka::malloc<TElem>(Allocator{}, static_cast<std::size_t>(getExtentProduct(extent)));
                auto deleter = [queue = std::move(queue)](TElem* ptr) mutable
                {
                    alpaka::enqueue(
                        queue,
                        [ptr]()
                        {
                            // free the memory
                            alpaka::free(Allocator{}, ptr);
                        });
                };

                return BufCpu<TElem, TDim, TIdx>(dev, memPtr, std::move(deleter), extent);
            }
        };
        //! The BufCpu stream-ordered memory allocation capability trait specialization.
        template<typename TDim>
        struct HasAsyncBufSupport<TDim, DevCpu> : public std::true_type
        {
        };

        //! The pinned/mapped memory allocation trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAllocMapped<TElem, TDim, TIdx, DevCpu>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocMappedBuf(DevCpu const& host, DevCpu const&, TExtent const& extent)
                -> BufCpu<TElem, TDim, TIdx>
            {
                // Allocate standard host memory.
                return allocBuf(host, extent);
            }
        };

        //! The BufCpu memory mapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Map<BufCpu<TElem, TDim, TIdx>, DevCpu>
        {
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
        //! The BufCpu memory unmapping trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unmap<BufCpu<TElem, TDim, TIdx>, DevCpu>
        {
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
        //! The BufCpu memory pinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Pin<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto pin(BufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(!isPinned(buf))
                {
                    if(buf.m_spBufCpuImpl->m_extentElements.prod() != 0)
                    {
                        // - cudaHostRegisterDefault:
                        //   See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                        // - cudaHostRegisterPortable:
                        //   The memory returned by this call will be considered as pinned memory by all CUDA contexts,
                        //   not just the one that performed the allocation.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                        {
                            using TApi = ApiCudaRt;
                            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                                TApi::hostRegister(
                                    const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                                    getExtentProduct(buf) * sizeof(Elem<BufCpu<TElem, TDim, TIdx>>),
                                    TApi::hostRegisterDefault),
                                TApi::errorHostMemoryAlreadyRegistered);
                        }
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                        {
                            using TApi = ApiHipRt;
                            ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                                TApi::hostRegister(
                                    const_cast<void*>(reinterpret_cast<void const*>(getPtrNative(buf))),
                                    getExtentProduct(buf) * sizeof(Elem<BufCpu<TElem, TDim, TIdx>>),
                                    TApi::hostRegisterDefault),
                                TApi::errorHostMemoryAlreadyRegistered);
                        }
#endif
                        buf.m_spBufCpuImpl->m_bPinned = true;
                    }
                }
            }
        };
        //! The BufCpu memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto unpin(BufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                alpaka::unpin(*buf.m_spBufCpuImpl.get());
            }
        };
        //! The BufCpuImpl memory unpinning trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct Unpin<alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto unpin(alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>& bufImpl) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                if(isPinned(bufImpl))
                {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    {
                        using TApi = ApiCudaRt;
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                            TApi::hostUnregister(const_cast<void*>(reinterpret_cast<void const*>(bufImpl.m_pMem))),
                            TApi::errorHostMemoryNotRegistered);
                    }
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
                    {
                        using TApi = ApiHipRt;
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
                            TApi::hostUnregister(const_cast<void*>(reinterpret_cast<void const*>(bufImpl.m_pMem))),
                            TApi::errorHostMemoryNotRegistered);
                    }
#endif
                    bufImpl.m_bPinned = false;
                }
            }
        };
        //! The BufCpu memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto isPinned(BufCpu<TElem, TDim, TIdx> const& buf) -> bool
            {
                return alpaka::isPinned(*buf.m_spBufCpuImpl.get());
            }
        };
        //! The BufCpuImpl memory pin state trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IsPinned<alpaka::detail::BufCpuImpl<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto isPinned(
                [[maybe_unused]] alpaka::detail::BufCpuImpl<TElem, TDim, TIdx> const& bufImpl) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                return bufImpl.m_bPinned;
            }
        };
        //! The BufCpu memory prepareForAsyncCopy trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct PrepareForAsyncCopy<BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto prepareForAsyncCopy([[maybe_unused]] BufCpu<TElem, TDim, TIdx>& buf) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // to optimize the data transfer performance between a cuda/hip device the cpu buffer has to be pinned,
                // for exclusive cpu use, no preparing is needed
                pin(buf);
            }
        };

        //! The BufCpu offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, BufCpu<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(BufCpu<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The BufCpu idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<BufCpu<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka

#include <alpaka/mem/buf/cpu/Copy.hpp>
#include <alpaka/mem/buf/cpu/Set.hpp>
