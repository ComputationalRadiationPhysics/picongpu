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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/DevCudaRt.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Cuda.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <memory>
#include <cassert>

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
                typename TSize>
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
                typename TSize>
            class BufCudaRt
            {
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
                    TSize const & pitchBytes,
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
                        std::is_same<TSize, size::Size<TExtent>>::value,
                        "The size type of TExtent and the TSize template parameter have to be identical!");
                }

            private:
                //-----------------------------------------------------------------------------
                //! Frees the shared buffer.
                ALPAKA_FN_HOST static auto freeBuffer(
                    TElem * memPtr,
                    dev::DevCudaRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device. \TODO: Is setting the current device before cudaFree required?
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));
                    // Free the buffer.
                    cudaFree(reinterpret_cast<void *>(memPtr));
                }

            public:
                dev::DevCudaRt m_dev;               // NOTE: The device has to be destructed after the memory pointer because it is required for destruction.
                vec::Vec<TDim, TSize> m_extentElements;
                std::shared_ptr<TElem> m_spMem;
                TSize m_pitchBytes;
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
                typename TSize>
            struct DevType<
                mem::buf::BufCudaRt<TElem, TDim, TSize>>
            {
                using type = dev::DevCudaRt;
            };
            //#############################################################################
            //! The BufCudaRt device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetDev<
                mem::buf::BufCudaRt<TElem, TDim, TSize>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufCudaRt<TElem, TDim, TSize> const & buf)
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
                typename TSize>
            struct DimType<
                mem::buf::BufCudaRt<TElem, TDim, TSize>>
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
                typename TSize>
            struct ElemType<
                mem::buf::BufCudaRt<TElem, TDim, TSize>>
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
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::buf::BufCudaRt<TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufCudaRt<TElem, TDim, TSize> const & extent)
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
                //! The BufCudaRt native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrNative<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> const & buf)
                    -> TElem const *
                    {
                        return buf.m_spMem.get();
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> & buf)
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
                    typename TSize>
                struct GetPtrDev<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> const & buf,
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
                        mem::buf::BufCudaRt<TElem, TDim, TSize> & buf,
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
                    typename TSize>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufCudaRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> const & buf)
                    -> TSize
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
                    typename T,
                    typename TSize>
                struct Alloc<
                    T,
                    dim::DimInt<1u>,
                    TSize,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCudaRt<T, dim::DimInt<1u>, TSize>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TSize>(sizeof(T)));

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
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << width
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<T, dim::DimInt<1u>, TSize>(
                                dev,
                                reinterpret_cast<T *>(memPtr),
                                static_cast<TSize>(widthBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The CUDA 2D memory allocation trait specialization.
                template<
                    typename T,
                    typename TSize>
                struct Alloc<
                    T,
                    dim::DimInt<2u>,
                    TSize,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCudaRt<T, dim::DimInt<2u>, TSize>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TSize>(sizeof(T)));
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
                        assert(pitchBytes>=widthBytes||(width*height)==0);

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << width
                            << " eh: " << height
                            << " ewb: " << widthBytes
                            << " ptr: " << memPtr
                            << " pitch: " << pitchBytes
                            << std::endl;
#endif
                        return
                            mem::buf::BufCudaRt<T, dim::DimInt<2u>, TSize>(
                                dev,
                                reinterpret_cast<T *>(memPtr),
                                static_cast<TSize>(pitchBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The CUDA 3D memory allocation trait specialization.
                template<
                    typename T,
                    typename TSize>
                struct Alloc<
                    T,
                    dim::DimInt<3u>,
                    TSize,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevCudaRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufCudaRt<T, dim::DimInt<3u>, TSize>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        cudaExtent const cudaExtentVal(
                            make_cudaExtent(
                                static_cast<std::size_t>(extent::getWidth(extent) * static_cast<TSize>(sizeof(T))),
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
                        std::cout << BOOST_CURRENT_FUNCTION
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
                            mem::buf::BufCudaRt<T, dim::DimInt<3u>, TSize>(
                                dev,
                                reinterpret_cast<T *>(cudaPitchedPtrVal.ptr),
                                static_cast<TSize>(cudaPitchedPtrVal.pitch),
                                extent);
                    }
                };
                //#############################################################################
                //! The BufCudaRt CUDA device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct Map<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> const & buf,
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
                    typename TSize>
                struct Unmap<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> const & buf,
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
                    typename TSize>
                struct Pin<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> &)
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
                    typename TSize>
                struct Unpin<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> &)
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
                    typename TSize>
                struct IsPinned<
                    mem::buf::BufCudaRt<TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufCudaRt<TElem, TDim, TSize> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // CUDA device memory is always pinned, it can not be swapped out.
                        return true;
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
                typename TIdx,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::buf::BufCudaRt<TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufCudaRt<TElem, TDim, TSize> const &)
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
            //! The BufCudaRt size type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TSize>
            struct SizeType<
                mem::buf::BufCudaRt<TElem, TDim, TSize>>
            {
                using type = TSize;
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
                    typename TSize>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
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
                                    extent::getProductOfExtent(buf) * sizeof(elem::Elem<BufCpu<TElem, TDim, TSize>>),
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
                    typename TSize>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
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
                    typename TSize>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim, TSize>,
                    dev::DevCudaRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TSize> const & buf,
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
                        mem::buf::BufCpu<TElem, TDim, TSize> & buf,
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
