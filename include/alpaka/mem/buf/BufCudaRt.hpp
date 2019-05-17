/* Copyright 2019 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Cuda.hpp>
#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <memory>

namespace alpaka
{
    namespace dev
    {
        class DevCudaRt;
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufCpu;
        }
    }
    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The CUDA memory buffer.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufCudaRt
            {
                static_assert(
                    !std::is_const<TElem>::value,
                    "The elem type of the buffer can not be const because the C++ Standard forbids containers of const elements!");
                static_assert(
                    !std::is_const<TIdx>::value,
                    "The idx type of the buffer can not be const!");
            private:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                template<
                    typename TExtent>
                ALPAKA_FN_HOST BufCudaRt(
                    dev::DevCudaRt const & dev,
                    TElem * const pMem,
                    TIdx const & pitchBytes,
                    TExtent const & extent) :
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_spMem(
                            pMem,
                            // NOTE: Because the BufCudaRt object can be copied and the original object could have been destroyed,
                            // a std::ref(m_dev) or a this pointer can not be bound to the callback because they are not always valid at time of destruction.
                            std::bind(&BufCudaRt::freeBuffer, std::placeholders::_1, m_dev)),
                        m_pitchBytes(pitchBytes)
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    static_assert(
                        TDim::value == dim::Dim<TExtent>::value,
                        "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");
                    static_assert(
                        std::is_same<TIdx, idx::Idx<TExtent>>::value,
                        "The idx type of TExtent and the TIdx template parameter have to be identical!");
                }

            private:
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                ALPAKA_FN_HOST static auto freeBuffer(
                    TElem * const memPtr,
                    dev::DevCudaRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
                    // Free the buffer.
                    ALPAKA_CUDA_RT_CHECK(
                      cudaFree(reinterpret_cast<void *>(memPtr)));
                }

            public:
                dev::DevCudaRt m_dev;               // NOTE: The device has to be destructed after the memory pointer because it is required for destruction.
                vec::Vec<TDim, TIdx> m_extentElements;
                std::shared_ptr<TElem> m_spMem;
                TIdx m_pitchBytes;
            };
        }
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DevType<
                mem::buf::BufCudaRt<TElem, TDim, TIdx>>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The BufCudaRt device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetDev<
                mem::buf::BufCudaRt<TElem, TDim, TIdx>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufCudaRt<TElem, TDim, TIdx> const & buf)
                -> dev::DevCudaRt
                {
                    return buf.m_dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt dimension getter trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DimType<
                mem::buf::BufCudaRt<TElem, TDim, TIdx>>
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
            //! The BufCudaRt memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct ElemType<
                mem::buf::BufCudaRt<TElem, TDim, TIdx>>
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
            //! The BufCudaRt extent get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                mem::buf::BufCudaRt<TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufCudaRt<TElem, TDim, TIdx> const & extent)
                -> TIdx
                {
                    return extent.m_extentElements[TIdxIntegralConst::value];
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
                //! The BufCudaRt native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrNative<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> const & buf)
                    -> TElem const *
                    {
                        return buf.m_spMem.get();
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> & buf)
                    -> TElem *
                    {
                        return buf.m_spMem.get();
                    }
                };
                //#############################################################################
                //! The BufCudaRt pointer on device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> const & buf,
                        dev::DevCudaRt const & dev)
                    -> TElem const *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return buf.m_spMem.get();
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> & buf,
                        dev::DevCudaRt const & dev)
                    -> TElem *
                    {
                        if(dev == dev::getDev(buf))
                        {
                            return buf.m_spMem.get();
                        }
                        else
                        {
                            throw std::runtime_error("The buffer is not accessible from the given device!");
                        }
                    }
                };
                //#############################################################################
                //! The BufCudaRt pitch get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> const & buf)
                    -> TIdx
                    {
                        return buf.m_pitchBytes;
                    }
                };
            }
        }
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The CUDA 1D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<1u>,
                    TIdx,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCudaRt<TElem, dim::DimInt<1u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * memPtr;
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMalloc(
                                &memPtr,
                                static_cast<std::size_t>(widthBytes)));

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << width
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<TElem, dim::DimInt<1u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                static_cast<TIdx>(widthBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The CUDA 2D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<2u>,
                    TIdx,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCudaRt<TElem, dim::DimInt<2u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));
                        auto const height(extent::getHeight(extent));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * memPtr;
                        std::size_t pitchBytes;
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMallocPitch(
                                &memPtr,
                                &pitchBytes,
                                static_cast<std::size_t>(widthBytes),
                                static_cast<std::size_t>(height)));
                        ALPAKA_ASSERT(pitchBytes >= static_cast<std::size_t>(widthBytes) || (width * height) == 0);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << width
                            << " eh: " << height
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << " pitch: " << pitchBytes
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<TElem, dim::DimInt<2u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                static_cast<TIdx>(pitchBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The CUDA 3D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<3u>,
                    TIdx,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCudaRt<TElem, dim::DimInt<3u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        cudaExtent const cudaExtentVal(
                            make_cudaExtent(
                                static_cast<std::size_t>(extent::getWidth(extent) * static_cast<TIdx>(sizeof(TElem))),
                                static_cast<std::size_t>(extent::getHeight(extent)),
                                static_cast<std::size_t>(extent::getDepth(extent))));

                        // Set the current device.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaSetDevice(
                                dev.m_iDevice));
                        // Allocate the buffer on this device.
                        cudaPitchedPtr cudaPitchedPtrVal;
                        ALPAKA_CUDA_RT_CHECK(
                            cudaMalloc3D(
                                &cudaPitchedPtrVal,
                                cudaExtentVal));


#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << __func__
                            << " ew: " << extent::getWidth(extent)
                            << " eh: " << cudaExtentVal.height
                            << " ed: " << cudaExtentVal.depth
                            << " ewb: " << cudaExtentVal.width
                            << " ptr: " << cudaPitchedPtrVal.ptr
                            << " pitch: " << cudaPitchedPtrVal.pitch
                            << " wb: " << cudaPitchedPtrVal.xsize
                            << " h: " << cudaPitchedPtrVal.ysize
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<TElem, dim::DimInt<3u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(cudaPitchedPtrVal.ptr),
                                static_cast<TIdx>(cudaPitchedPtrVal.pitch),
                                extent);
                    }
                };
                //#############################################################################
                //! The BufCudaRt CUDA device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> const & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Mapping memory from one CUDA device into an other CUDA device not implemented!");
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCudaRt CUDA device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> const & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Unmapping memory mapped from one CUDA device into an other CUDA device not implemented!");
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
                //#############################################################################
                //! The BufCudaRt memory pinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Pin<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufCudaRt memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unpin<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufCudaRt memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct IsPinned<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always pinned, it can not be swapped out.
                        return true;
                    }
                };
                //#############################################################################
                //! The BufCudaRt memory prepareForAsyncCopy trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct PrepareForAsyncCopy<
                    mem::buf::BufCudaRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto prepareForAsyncCopy(
                        mem::buf::BufCudaRt<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always ready for async copy
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
            //! The BufCudaRt offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                mem::buf::BufCudaRt<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufCudaRt<TElem, TDim, TIdx> const &)
                -> TIdx
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The BufCudaRt idx type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct IdxType<
                mem::buf::BufCudaRt<TElem, TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufCpu.
    namespace mem
    {
        namespace buf
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu CUDA device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // cudaHostRegisterMapped:
                            //   Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                            //   This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                            ALPAKA_CUDA_RT_CHECK(
                                cudaHostRegister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                    extent::getExtentProduct(buf) * sizeof(elem::Elem<BufCpu<TElem, TDim, TIdx>>),
                                    cudaHostRegisterMapped));
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCpu CUDA device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevCudaRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.
                            // \FIXME: If the memory has separately been pinned before we destroy the pinning state.
                            ALPAKA_CUDA_RT_CHECK(
                                cudaHostUnregister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf)))));
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
            }
        }
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufCpu pointer on CUDA device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> const & buf,
                        dev::DevCudaRt const &)
                    -> TElem const *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_CUDA_RT_CHECK(
                            cudaHostGetDevicePointer(
                                &pDev,
                                const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                0));
                        return pDev;
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevCudaRt const &)
                    -> TElem *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_CUDA_RT_CHECK(
                            cudaHostGetDevicePointer(
                                &pDev,
                                mem::view::getPtrNative(buf),
                                0));
                        return pDev;
                    }
                };
            }
        }
    }
}

#include <alpaka/mem/buf/cuda/Copy.hpp>
#include <alpaka/mem/buf/cuda/Set.hpp>

#endif
