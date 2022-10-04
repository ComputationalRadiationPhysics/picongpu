/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/dim/Traits.hpp>
#    include <alpaka/mem/buf/BufCpu.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/mem/view/Accessor.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <CL/sycl.hpp>

#    include <memory>
#    include <type_traits>

namespace alpaka::experimental
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl
    {
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

    public:
        //! Constructor
        template<typename TExtent>
        BufGenericSycl(TDev const& dev, sycl::buffer<TElem, TDim::value> buffer, TExtent const& extent)
            : m_dev{dev}
            , m_extentElements{getExtentVecEnd<TDim>(extent)}
            , m_buffer{buffer}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");

            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        TDev m_dev;
        Vec<TDim, TIdx> m_extentElements;
        sycl::buffer<TElem, TDim::value> m_buffer;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    //! The BufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct DevType<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TDev;
    };

    //! The BufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetDev<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static auto getDev(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf)
        {
            return buf.m_dev;
        }
    };

    //! The BufGenericSycl dimension getter trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct DimType<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TDim;
    };

    //! The BufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct ElemType<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TElem;
    };

    //! The BufGenericSycl extent get trait specialization.
    template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetExtent<TIdxIntegralConst, experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static_assert(TDim::value > TIdxIntegralConst::value, "Requested dimension out of bounds");

        static auto getExtent(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev> const& buf) -> TIdx
        {
            return buf.m_extentElements[TIdxIntegralConst::value];
        }
    };

    //! The BufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetPtrNative<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static_assert(
            !sizeof(TElem),
            "Accessing device-side pointers on the host is not supported by the SYCL back-end");

        static auto getPtrNative(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev> const&) -> TElem const*
        {
            return nullptr;
        }

        static auto getPtrNative(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>&) -> TElem*
        {
            return nullptr;
        }
    };

    //! The BufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetPtrDev<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>, TDev>
    {
        static_assert(
            !sizeof(TElem),
            "Accessing device-side pointers on the host is not supported by the SYCL back-end");

        static auto getPtrDev(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev> const&, TDev const&)
            -> TElem const*
        {
            return nullptr;
        }

        static auto getPtrDev(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>&, TDev const&) -> TElem*
        {
            return nullptr;
        }
    };

    //! The SYCL memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct BufAlloc<TElem, TDim, TIdx, experimental::DevGenericSycl<TPltf>>
    {
        template<typename TExtent>
        static auto allocBuf(experimental::DevGenericSycl<TPltf> const& dev, TExtent const& ext)
            -> experimental::BufGenericSycl<TElem, TDim, TIdx, experimental::DevGenericSycl<TPltf>>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            if constexpr(TDim::value == 0 || TDim::value == 1)
            {
                auto const width = getWidth(ext);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << '\n';
#    endif

                auto const range = sycl::range<1>{width};
                return {dev, sycl::buffer<TElem, 1>{range}, ext};
            }
            else if constexpr(TDim::value == 2)
            {
                auto const width = getWidth(ext);
                auto const height = getHeight(ext);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ewb: " << widthBytes
                          << " pitch: " << widthBytes << '\n';
#    endif

                auto const range = sycl::range<2>{width, height};
                return {dev, sycl::buffer<TElem, 2>{range}, ext};
            }
            else if constexpr(TDim::value == 3)
            {
                auto const width = getWidth(ext);
                auto const height = getHeight(ext);
                auto const depth = getDepth(ext);

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ed: " << depth
                          << " ewb: " << widthBytes << " pitch: " << widthBytes << '\n';
#    endif

                auto const range = sycl::range<3>{width, height, depth};
                return {dev, sycl::buffer<TElem, 3>{range}, ext};
            }
        }
    };

    //! The BufGenericSycl offset get trait specialization.
    template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx, typename TDev>
    struct GetOffset<TIdxIntegralConst, experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        static auto getOffset(experimental::BufGenericSycl<TElem, TDim, TIdx, TDev> const&) -> TIdx
        {
            return 0u;
        }
    };

    //! The BufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    struct IdxType<experimental::BufGenericSycl<TElem, TDim, TIdx, TDev>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on SYCL device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPltf>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, experimental::DevGenericSycl<TPltf>>
    {
        static_assert(!sizeof(TElem), "Accessing host pointers on the device is not supported by the SYCL back-end");

        static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const&, experimental::DevGenericSycl<TPltf> const&)
            -> TElem const*
        {
            return nullptr;
        }

        static auto getPtrDev(BufCpu<TElem, TDim, TIdx>&, experimental::DevGenericSycl<TPltf> const&) -> TElem*
        {
            return nullptr;
        }
    };
} // namespace alpaka::trait

#    include <alpaka/mem/buf/sycl/Accessor.hpp>
#    include <alpaka/mem/buf/sycl/Copy.hpp>
#    include <alpaka/mem/buf/sycl/Set.hpp>

#endif
