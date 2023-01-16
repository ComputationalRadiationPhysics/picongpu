/* Copyright 2022 Jeffrey Kelling, Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera,
 *                Bernhard Manfred Gruber, Antonio Di Pilato, Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/dev/DevOacc.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/mem/view/ViewAccessOps.hpp>
#    include <alpaka/queue/QueueOaccBlocking.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <openacc.h>

#    include <memory>

namespace alpaka
{
    class DevOacc;

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    namespace oacc::detail
    {
        //! The OpenACC memory buffer detail.
        template<typename TElem, typename TDim, typename TIdx>
        class BufOaccImpl
        {
            static_assert(
                !std::is_const_v<TElem>,
                "The elem type of the buffer can not be const because the C++ Standard forbids containers of "
                "const elements!");
            static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        public:
            //! Constructor
            template<typename TExtent>
            ALPAKA_FN_HOST BufOaccImpl(DevOacc const& dev, TElem* const pMem, TExtent const& extent) noexcept
                : m_dev(dev)
                , m_extentElements(getExtentVecEnd<TDim>(extent))
                , m_pMem(pMem)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    TDim::value == alpaka::Dim<TExtent>::value,
                    "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to "
                    "be identical!");
                static_assert(
                    std::is_same<TIdx, Idx<TExtent>>::value,
                    "The idx type of TExtent and the TIdx template parameter have to be identical!");
            }

        public:
            DevOacc m_dev;
            Vec<TDim, TIdx> m_extentElements;
            TElem* m_pMem;

            BufOaccImpl(BufOaccImpl&&) = delete;
            BufOaccImpl& operator=(BufOaccImpl&&) = delete;
            ~BufOaccImpl()
            {
                m_dev.makeCurrent();
                acc_free(m_pMem);
            }
        };
    } // namespace oacc::detail

    template<typename TElem, typename TDim, typename TIdx>
    class BufOacc : public internal::ViewAccessOps<BufOacc<TElem, TDim, TIdx>>
    {
    public:
        //! Constructor
        template<typename TExtent>
        ALPAKA_FN_HOST BufOacc(DevOacc const& dev, TElem* const pMem, TExtent const& extent)
            : m_spBufImpl(std::make_shared<oacc::detail::BufOaccImpl<TElem, TDim, TIdx>>(dev, pMem, extent))
        {
        }

        oacc::detail::BufOaccImpl<TElem, TDim, TIdx>& operator*()
        {
            return *m_spBufImpl;
        }
        oacc::detail::BufOaccImpl<TElem, TDim, TIdx> const& operator*() const
        {
            return *m_spBufImpl;
        }

        Vec<TDim, TIdx> const& extentElements() const
        {
            return m_spBufImpl->m_extentElements;
        }

    private:
        std::shared_ptr<oacc::detail::BufOaccImpl<TElem, TDim, TIdx>> m_spBufImpl;
    };

    namespace trait
    {
        //! The BufOacc device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufOacc<TElem, TDim, TIdx>>
        {
            using type = DevOacc;
        };
        //! The BufOacc device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<BufOacc<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(BufOacc<TElem, TDim, TIdx> const& buf) -> DevOacc
            {
                return (*buf).m_dev;
            }
        };

        //! The BufOacc dimension getter trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<BufOacc<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The BufOacc memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<BufOacc<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        //! The BufOacc extent get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetExtent<
            TIdxIntegralConst,
            BufOacc<TElem, TDim, TIdx>,
            typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
        {
            ALPAKA_FN_HOST static auto getExtent(BufOacc<TElem, TDim, TIdx> const& extent) -> TIdx
            {
                return extent.extentElements()[TIdxIntegralConst::value];
            }
        };

        //! The BufOacc native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<BufOacc<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(BufOacc<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return (*buf).m_pMem;
            }
            ALPAKA_FN_HOST static auto getPtrNative(BufOacc<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return (*buf).m_pMem;
            }
        };
        //! The BufOacc pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufOacc<TElem, TDim, TIdx>, DevOacc>
        {
            ALPAKA_FN_HOST static auto getPtrDev(BufOacc<TElem, TDim, TIdx> const& buf, DevOacc const& dev)
                -> TElem const*
            {
                if(dev == getDev(buf))
                {
                    return *buf.m_pMem;
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
            ALPAKA_FN_HOST static auto getPtrDev(BufOacc<TElem, TDim, TIdx>& buf, DevOacc const& dev) -> TElem*
            {
                if(dev == getDev(buf))
                {
                    return *buf.m_pMem;
                }
                else
                {
                    throw std::runtime_error("The buffer is not accessible from the given device!");
                }
            }
        };

        //! The BufOacc 1D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct BufAlloc<TElem, DimInt<1u>, TIdx, DevOacc>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevOacc const& dev, TExtent const& extent)
                -> BufOacc<TElem, DimInt<1u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const width(getWidth(extent));
                auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                dev.makeCurrent();
                void* memPtr = acc_malloc(static_cast<std::size_t>(widthBytes));

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << " ptr: " << memPtr
                          << " device: " << dev.getNativeHandle() << std::endl;
#    endif
                return BufOacc<TElem, DimInt<1u>, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), extent);
            }
        };

        //! The BufOacc nD memory allocation trait specialization. \todo Add pitch
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAlloc<TElem, TDim, TIdx, DevOacc>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevOacc const& dev, TExtent const& extent)
                -> BufOacc<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                const std::size_t size = static_cast<std::size_t>(getExtentVec(extent).prod()) * sizeof(TElem);

                dev.makeCurrent();
                void* memPtr = acc_malloc(size);
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << "alloc'd " << TDim::value << "D device ptr: " << memPtr << " on device "
                          << dev.getNativeHandle() << " size: " << size << std::endl;
#    endif
                return BufOacc<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), extent);
            }
        };

        //! The BufOacc offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, BufOacc<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(BufOacc<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The BufOacc idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<BufOacc<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The BufCpu pointer on OpenAcc device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevOacc>
        {
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const&, DevOacc const&) -> TElem const*
            {
                throw std::runtime_error("Mapping host memory to OpenACC device not implemented!");
            }
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>&, DevOacc const&) -> TElem*
            {
                throw std::runtime_error("Mapping host memory to OpenACC device not implemented!");
            }
        };
    } // namespace trait
} // namespace alpaka

#    include <alpaka/mem/buf/oacc/Copy.hpp>
#    include <alpaka/mem/buf/oacc/Set.hpp>

#endif
