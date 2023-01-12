/* Copyright 2022 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Andrea Bocci, Jan Stephan,
 * Bernhard Manfred Gruber, Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/core/Cuda.hpp>
#    include <alpaka/core/Hip.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/mem/view/ViewAccessOps.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <functional>
#    include <memory>
#    include <type_traits>

namespace alpaka
{
    // Forward declarations.
    struct ApiCudaRt;
    struct ApiHipRt;

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    //! The CUDA/HIP memory buffer.
    template<typename TApi, typename TElem, typename TDim, typename TIdx>
    class BufUniformCudaHipRt : public internal::ViewAccessOps<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
    {
    public:
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        //! Constructor
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufUniformCudaHipRt(
            DevUniformCudaHipRt<TApi> const& dev,
            TElem* const pMem,
            Deleter deleter,
            TIdx const& pitchBytes,
            TExtent const& extent)
            : m_dev(dev)
            , m_extentElements(getExtentVecEnd<TDim>(extent))
            , m_spMem(pMem, std::move(deleter))
            , m_pitchBytes(pitchBytes)
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == alpaka::Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");
            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

    public:
        DevUniformCudaHipRt<TApi> m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_spMem;
        TIdx m_pitchBytes;
    };

    namespace trait
    {
        //! The BufUniformCudaHipRt device type trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct DevType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The BufUniformCudaHipRt device get trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct GetDev<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
                -> DevUniformCudaHipRt<TApi>
            {
                return buf.m_dev;
            }
        };

        //! The BufUniformCudaHipRt dimension getter trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct DimType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The BufUniformCudaHipRt memory element type get trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct ElemType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        //! The BufUniformCudaHipRt extent get trait specialization.
        template<typename TApi, typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetExtent<
            TIdxIntegralConst,
            BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>,
            std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
        {
            ALPAKA_FN_HOST static auto getExtent(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& extent) -> TIdx
            {
                return extent.m_extentElements[TIdxIntegralConst::value];
            }
        };

        //! The BufUniformCudaHipRt native pointer get trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf)
                -> TElem const*
            {
                return buf.m_spMem.get();
            }
            ALPAKA_FN_HOST static auto getPtrNative(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>& buf) -> TElem*
            {
                return buf.m_spMem.get();
            }
        };

        //! The BufUniformCudaHipRt pointer on device get trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getPtrDev(
                BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf,
                DevUniformCudaHipRt<TApi> const& dev) -> TElem const*
            {
                if(dev == getDev(buf))
                {
                    return buf.m_spMem.get();
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
            ALPAKA_FN_HOST static auto getPtrDev(
                BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>& buf,
                DevUniformCudaHipRt<TApi> const& dev) -> TElem*
            {
                if(dev == getDev(buf))
                {
                    return buf.m_spMem.get();
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
        };

        //! The BufUniformCudaHipRt pitch get trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct GetPitchBytes<DimInt<TDim::value - 1u>, BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPitchBytes(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const& buf) -> TIdx
            {
                return buf.m_pitchBytes; // TODO(bgruber): is this even correct? This reports the pitch for the TDim -
                                         // 1 dimension, but CUDA's pitch is always the row pitch, independently of the
                                         // buffer's dimensions.
            }
        };

        //! The CUDA/HIP memory allocation trait specialization.
        template<typename TApi, typename TElem, typename Dim, typename TIdx>
        struct BufAlloc<TElem, Dim, TIdx, DevUniformCudaHipRt<TApi>>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevUniformCudaHipRt<TApi> const& dev, TExtent const& extent)
                -> BufUniformCudaHipRt<TApi, TElem, Dim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

                void* memPtr = nullptr;
                std::size_t pitchBytes = 0u;
                if(getExtentProduct(extent) != 0)
                {
                    if constexpr(Dim::value <= 1)
                    {
                        pitchBytes = static_cast<std::size_t>(getWidth(extent)) * sizeof(TElem);
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::malloc(&memPtr, pitchBytes));
                    }
                    else if constexpr(Dim::value == 2)
                    {
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::mallocPitch(
                            &memPtr,
                            &pitchBytes,
                            static_cast<std::size_t>(getWidth(extent)) * sizeof(TElem),
                            static_cast<std::size_t>(getHeight(extent))));
                    }
                    else if constexpr(Dim::value == 3)
                    {
                        typename TApi::Extent_t const extentVal = TApi::makeExtent(
                            static_cast<std::size_t>(getWidth(extent)) * sizeof(TElem),
                            static_cast<std::size_t>(getHeight(extent)),
                            static_cast<std::size_t>(getDepth(extent)));
                        typename TApi::PitchedPtr_t pitchedPtrVal;
                        pitchedPtrVal.ptr = nullptr;
                        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::malloc3D(&pitchedPtrVal, extentVal));
                        memPtr = pitchedPtrVal.ptr;
                        pitchBytes = pitchedPtrVal.pitch;
                    }
                }
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__;
                if constexpr(Dim::value >= 1)
                    std::cout << " ew: " << getWidth(extent);
                if constexpr(Dim::value >= 2)
                    std::cout << " eh: " << getHeight(extent);
                if constexpr(Dim::value >= 3)
                    std::cout << " ed: " << getDepth(extent);
                std::cout << " ptr: " << memPtr << " pitch: " << pitchBytes << std::endl;
#    endif
                return {
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    [](TElem* ptr) { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::free(ptr)); },
                    static_cast<TIdx>(pitchBytes),
                    extent};
            }
        };

        //! The CUDA/HIP stream-ordered memory allocation trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct AsyncBufAlloc<TElem, TDim, TIdx, DevUniformCudaHipRt<TApi>>
        {
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            static_assert(
                std::is_same_v<TApi, ApiCudaRt> && TApi::version >= BOOST_VERSION_NUMBER(11, 2, 0),
                "Support for stream-ordered memory buffers requires CUDA 11.2 or higher.");
#    endif
#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
            static_assert(
                !std::is_same_v<TApi, ApiHipRt>,
                "HIP devices do not support stream-ordered memory buffers.");
#    endif
            static_assert(
                TDim::value <= 1,
                "CUDA/HIP devices support only one-dimensional stream-ordered memory buffers.");

            template<typename TQueue, typename TExtent>
            ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, [[maybe_unused]] TExtent const& extent)
                -> BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(TDim::value == Dim<TExtent>::value, "extent must have the same dimension as the buffer");
                auto const width = getWidth(extent);

                auto const& dev = getDev(queue);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));
                void* memPtr = nullptr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::mallocAsync(
                    &memPtr,
                    static_cast<std::size_t>(width) * sizeof(TElem),
                    queue.getNativeHandle()));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " ptr: " << memPtr << std::endl;
#    endif
                return BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>(
                    dev,
                    reinterpret_cast<TElem*>(memPtr),
                    [queue = std::move(queue)](TElem* ptr)
                    { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::freeAsync(ptr, queue.getNativeHandle())); },
                    width * static_cast<TIdx>(sizeof(TElem)),
                    extent);
            }
        };

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        //! The CUDA/HIP stream-ordered memory allocation capability trait specialization.
        template<typename TApi, typename TDim>
        struct HasAsyncBufSupport<TDim, DevUniformCudaHipRt<TApi>>
            : std::bool_constant<
                  std::is_same_v<TApi, ApiCudaRt> && TApi::version >= BOOST_VERSION_NUMBER(11, 2, 0)
                  && TDim::value <= 1>
        {
        };
#    endif

        //! The pinned/mapped memory allocation trait specialization for the CUDA/HIP devices.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct BufAllocMapped<PltfUniformCudaHipRt<TApi>, TElem, TDim, TIdx>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocMappedBuf(DevCpu const& host, TExtent const& extent)
                -> BufCpu<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // Allocate CUDA/HIP page-locked memory on the host, mapped into the CUDA/HIP address space and
                // accessible to all CUDA/HIP devices.
                TElem* memPtr = nullptr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostMalloc(
                    reinterpret_cast<void**>(&memPtr),
                    sizeof(TElem) * static_cast<std::size_t>(getExtentProduct(extent)),
                    TApi::hostMallocMapped | TApi::hostMallocPortable));
                auto deleter = [](TElem* ptr) { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::hostFree(ptr)); };

                return BufCpu<TElem, TDim, TIdx>(host, memPtr, std::move(deleter), extent);
            }
        };

        //! The pinned/mapped memory allocation capability trait specialization.
        template<typename TApi>
        struct HasMappedBufSupport<PltfUniformCudaHipRt<TApi>> : public std::true_type
        {
        };

        //! The BufUniformCudaHipRt offset get trait specialization.
        template<typename TApi, typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(BufUniformCudaHipRt<TApi, TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The BufUniformCudaHipRt idx type trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct IdxType<BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The BufCpu pointer on CUDA/HIP device get trait specialization.
        template<typename TApi, typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getPtrDev(
                BufCpu<TElem, TDim, TIdx> const& buf,
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
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevUniformCudaHipRt<TApi> const&)
                -> TElem*
            {
                // TODO: Check if the memory is mapped at all!
                TElem* pDev(nullptr);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::hostGetDevicePointer(&pDev, getPtrNative(buf), 0));

                return pDev;
            }
        };
    } // namespace trait
} // namespace alpaka

#    include <alpaka/mem/buf/uniformCudaHip/Copy.hpp>
#    include <alpaka/mem/buf/uniformCudaHip/Set.hpp>

#endif
