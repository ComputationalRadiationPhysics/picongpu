/**
* \file
* Copyright 2014-2018 Benjamin Worpitz, Alexander Matthes
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

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Hip.hpp>
#include <alpaka/dev/DevHipRt.hpp>
#include <alpaka/vec/Vec.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <memory>

namespace alpaka
{
    namespace dev
    {
        class DevHipRt;
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
            //! The HIP memory buffer.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufHipRt
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
                ALPAKA_FN_HOST BufHipRt(
                    dev::DevHipRt const & dev,
                    TElem * const pMem,
                    TIdx const & pitchBytes,
                    TExtent const & extent) :
                        m_dev(dev),
                        m_extentElements(extent::getExtentVecEnd<TDim>(extent)),
                        m_spMem(
                            pMem,
                            // NOTE: Because the BufHipRt object can be copied and the original object could have been destroyed,
                            // a std::ref(m_dev) or a this pointer can not be bound to the callback because they are not always valid at time of destruction.
                            std::bind(&BufHipRt::freeBuffer, std::placeholders::_1, m_dev)),
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
                    dev::DevHipRt const & dev)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_HIP_RT_CHECK(
                        hipSetDevice(
                            dev.m_iDevice));
                    // Free the buffer.
                    ALPAKA_HIP_RT_CHECK(
                        hipFree(reinterpret_cast<void *>(memPtr)));
                }

            public:
                dev::DevHipRt m_dev;               // NOTE: The device has to be destructed after the memory pointer because it is required for destruction.
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
            //! The BufHipRt device type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DevType<
                mem::buf::BufHipRt<TElem, TDim, TIdx>>
            {
                using type = dev::DevHipRt;
            };
            //#############################################################################
            //! The BufHipRt device get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetDev<
                mem::buf::BufHipRt<TElem, TDim, TIdx>>
            {
                ALPAKA_FN_HOST static auto getDev(
                    mem::buf::BufHipRt<TElem, TDim, TIdx> const & buf)
                -> dev::DevHipRt
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
            //! The BufHipRt dimension getter trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct DimType<
                mem::buf::BufHipRt<TElem, TDim, TIdx>>
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
            //! The BufHipRt memory element type get trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct ElemType<
                mem::buf::BufHipRt<TElem, TDim, TIdx>>
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
            //! The BufHipRt extent get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetExtent<
                TIdxIntegralConst,
                mem::buf::BufHipRt<TElem, TDim, TIdx>,
                typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    mem::buf::BufHipRt<TElem, TDim, TIdx> const & extent)
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
                //! The BufHipRt native pointer get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrNative<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> const & buf)
                    -> TElem const *
                    {
                        return buf.m_spMem.get();
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> & buf)
                    -> TElem *
                    {
                        return buf.m_spMem.get();
                    }
                };
                //#############################################################################
                //! The BufHipRt pointer on device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> const & buf,
                        dev::DevHipRt const & dev)
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
                        mem::buf::BufHipRt<TElem, TDim, TIdx> & buf,
                        dev::DevHipRt const & dev)
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
                //! The BufHipRt pitch get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPitchBytes<
                    dim::DimInt<TDim::value - 1u>,
                    mem::buf::BufHipRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> const & buf)
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
                //! The HIP 1D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<1u>,
                    TIdx,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevHipRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufHipRt<TElem, dim::DimInt<1u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                dev.m_iDevice));
                        // Allocate the buffer on this device.
                        void * memPtr;
                        ALPAKA_HIP_RT_CHECK(
                            hipMalloc(
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
                            mem::buf::BufHipRt<TElem, dim::DimInt<1u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                static_cast<TIdx>(widthBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The HIP 2D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<2u>,
                    TIdx,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevHipRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufHipRt<TElem, dim::DimInt<2u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        auto const width(extent::getWidth(extent));
                        auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));
                        auto const height(extent::getHeight(extent));

                        void * memPtr = nullptr;
                        std::size_t pitchBytes = widthBytes;

                        //FIXME: hcc cannot handle zero-size input (throws Unknown Error)
                        if(width!=0 && height!=0) {

                            // Set the current device.
                            ALPAKA_HIP_RT_CHECK(
                                hipSetDevice(
                                    dev.m_iDevice));


                            // Allocate the buffer on this device.
                            ALPAKA_HIP_RT_CHECK(
                                hipMallocPitch(
                                    &memPtr,
                                    &pitchBytes,
                                    static_cast<std::size_t>(widthBytes),
                                    static_cast<std::size_t>(height)));
                            ALPAKA_ASSERT(pitchBytes >= static_cast<std::size_t>(widthBytes) || (width * height) == 0);
                        }

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
                            mem::buf::BufHipRt<TElem, dim::DimInt<2u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(memPtr),
                                static_cast<TIdx>(pitchBytes),
                                extent);
                    }
                };
                //#############################################################################
                //! The HIP 3D memory allocation trait specialization.
                template<
                    typename TElem,
                    typename TIdx>
                struct Alloc<
                    TElem,
                    dim::DimInt<3u>,
                    TIdx,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent>
                    ALPAKA_FN_HOST static auto alloc(
                        dev::DevHipRt const & dev,
                        TExtent const & extent)
                    -> mem::buf::BufHipRt<TElem, dim::DimInt<3u>, TIdx>
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        hipExtent const hipExtentVal(
                            make_hipExtent(
                                static_cast<std::size_t>(extent::getWidth(extent) * static_cast<TIdx>(sizeof(TElem))),
                                static_cast<std::size_t>(extent::getHeight(extent)),
                                static_cast<std::size_t>(extent::getDepth(extent))));

                        hipPitchedPtr hipPitchedPtrVal = {0};

                        //FIXME: hcc cannot handle zero-size input
                        if(hipExtentVal.width!=0
                           && hipExtentVal.height!=0
                           && hipExtentVal.depth!=0) {

                            // Set the current device.
                            ALPAKA_HIP_RT_CHECK(
                                hipSetDevice(
                                    dev.m_iDevice));
                            // Allocate the buffer on this device.
                            ALPAKA_HIP_RT_CHECK(
                                hipMalloc3D(
                                    &hipPitchedPtrVal,
                                    hipExtentVal));
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        std::cout << BOOST_CURRENT_FUNCTION
                            << " ew: " << extent::getWidth(extent)
                            << " eh: " << hipExtentVal.height
                            << " ed: " << hipExtentVal.depth
                            << " ewb: " << hipExtentVal.width
                            << " ptr: " << hipPitchedPtrVal.ptr
                            << " pitch: " << hipPitchedPtrVal.pitch
                            << " wb: " << hipPitchedPtrVal.xsize
                            << " h: " << hipPitchedPtrVal.ysize
                            << std::endl;
#endif
                        return
                            mem::buf::BufHipRt<TElem, dim::DimInt<3u>, TIdx>(
                                dev,
                                reinterpret_cast<TElem *>(hipPitchedPtrVal.ptr),
                                static_cast<TIdx>(hipPitchedPtrVal.pitch),
                                extent);
                    }
                };
                //#############################################################################
                //! The BufHipRt HIP device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> const & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Mapping memory from one HIP device into an other HIP device not implemented!");
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufHipRt HIP device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> const & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            throw std::runtime_error("Unmapping memory mapped from one HIP device into an other HIP device not implemented!");
                        }
                        // If it is already the same device, nothing has to be unmapped.
                    }
                };
                //#############################################################################
                //! The BufHipRt memory pinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Pin<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto pin(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufHipRt memory unpinning trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unpin<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unpin(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always pinned, it can not be swapped out.
                    }
                };
                //#############################################################################
                //! The BufHipRt memory pin state trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct IsPinned<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto isPinned(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> const &)
                    -> bool
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always pinned, it can not be swapped out.
                        return true;
                    }
                };
                //#############################################################################
                //! The BufHipRt memory prepareForAsyncCopy trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct PrepareForAsyncCopy<
                    mem::buf::BufHipRt<TElem, TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto prepareForAsyncCopy(
                        mem::buf::BufHipRt<TElem, TDim, TIdx> &)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // HIP device memory is always ready for async copy
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
            //! The BufHipRt offset get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TElem,
                typename TDim,
                typename TIdx>
            struct GetOffset<
                TIdxIntegralConst,
                mem::buf::BufHipRt<TElem, TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                   mem::buf::BufHipRt<TElem, TDim, TIdx> const &)
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
            //! The BufHipRt idx type trait specialization.
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            struct IdxType<
                mem::buf::BufHipRt<TElem, TDim, TIdx>>
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
                //! The BufCpu HIP device memory mapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Map<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto map(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // hipHostRegisterMapped:
                            //   Maps the allocation into the HIP address space.The device pointer to the memory may be obtained by calling hipHostGetDevicePointer().
                            //   This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                            ALPAKA_HIP_RT_CHECK(
                                hipHostRegister(
                                    const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                    extent::getExtentProduct(buf) * sizeof(elem::Elem<BufCpu<TElem, TDim, TIdx>>),
                                    hipHostRegisterMapped));
                        }
                        // If it is already the same device, nothing has to be mapped.
                    }
                };
                //#############################################################################
                //! The BufCpu HIP device memory unmapping trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct Unmap<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto unmap(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevHipRt const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        if(dev::getDev(buf) != dev)
                        {
                            // Unmaps the memory range whose base address is specified by ptr, and makes it pageable again.
                            // \FIXME: If the memory has separately been pinned before we destroy the pinning state.
                            ALPAKA_HIP_RT_CHECK(
                                hipHostUnregister(
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
                //! The BufCpu pointer on HIP device get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct GetPtrDev<
                    mem::buf::BufCpu<TElem, TDim, TIdx>,
                    dev::DevHipRt>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> const & buf,
                        dev::DevHipRt const &)
                    -> TElem const *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_HIP_RT_CHECK(
                            hipHostGetDevicePointer(
                                &pDev,
                                const_cast<void *>(reinterpret_cast<void const *>(mem::view::getPtrNative(buf))),
                                0));
                        return pDev;
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrDev(
                        mem::buf::BufCpu<TElem, TDim, TIdx> & buf,
                        dev::DevHipRt const &)
                    -> TElem *
                    {
                        // TODO: Check if the memory is mapped at all!
                        TElem * pDev(nullptr);
                        ALPAKA_HIP_RT_CHECK(
                            hipHostGetDevicePointer(
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

#include <alpaka/mem/buf/hip/Copy.hpp>
#include <alpaka/mem/buf/hip/Set.hpp>

#endif
