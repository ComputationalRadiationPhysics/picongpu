/* Copyright 2022 Alexander Matthes, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Assert.hpp>
#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/mem/view/ViewAccessOps.hpp>
#    include <alpaka/queue/QueueOmp5Blocking.hpp>
#    include <alpaka/vec/Vec.hpp>

#    include <omp.h>

#    include <memory>

namespace alpaka
{
    class DevOmp5;

    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    namespace detail
    {
        //! The OMP5 memory buffer detail.
        template<typename TElem, typename TDim, typename TIdx>
        class BufOmp5Impl
        {
            static_assert(
                !std::is_const_v<TElem>,
                "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
                "elements!");
            static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        public:
            //! Constructor
            template<typename TExtent>
            ALPAKA_FN_HOST BufOmp5Impl(DevOmp5 const& dev, TElem* const pMem, TExtent const& extent) noexcept
                : m_dev(dev)
                , m_extentElements(getExtentVecEnd<TDim>(extent))
                , m_pMem(pMem)
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                static_assert(
                    TDim::value == alpaka::Dim<TExtent>::value,
                    "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                    "identical!");
                static_assert(
                    std::is_same<TIdx, Idx<TExtent>>::value,
                    "The idx type of TExtent and the TIdx template parameter have to be identical!");
            }

        public:
            DevOmp5 m_dev;
            Vec<TDim, TIdx> m_extentElements;
            TElem* m_pMem;

            BufOmp5Impl(BufOmp5Impl&&) = delete;
            auto operator=(BufOmp5Impl&&) -> BufOmp5Impl& = delete;
            ~BufOmp5Impl()
            {
                omp_target_free(m_pMem, m_dev.getNativeHandle());
            }
        };
    } // namespace detail

    template<typename TElem, typename TDim, typename TIdx>
    class BufOmp5 : public internal::ViewAccessOps<BufOmp5<TElem, TDim, TIdx>>
    {
    public:
        //! Constructor
        template<typename TExtent>
        ALPAKA_FN_HOST BufOmp5(DevOmp5 const& dev, TElem* const pMem, TExtent const& extent)
            : m_spBufImpl(std::make_shared<detail::BufOmp5Impl<TElem, TDim, TIdx>>(dev, pMem, extent))
        {
        }

        auto operator*() -> detail::BufOmp5Impl<TElem, TDim, TIdx>&
        {
            return *m_spBufImpl;
        }
        auto operator*() const -> detail::BufOmp5Impl<TElem, TDim, TIdx> const&
        {
            return *m_spBufImpl;
        }

        inline auto extentElements() const -> Vec<TDim, TIdx> const&
        {
            return m_spBufImpl->m_extentElements;
        }

    private:
        std::shared_ptr<detail::BufOmp5Impl<TElem, TDim, TIdx>> m_spBufImpl;
    };

    namespace trait
    {
        //! The BufOmp5 device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufOmp5<TElem, TDim, TIdx>>
        {
            using type = DevOmp5;
        };
        //! The BufOmp5 device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetDev<BufOmp5<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getDev(BufOmp5<TElem, TDim, TIdx> const& buf) -> DevOmp5
            {
                return (*buf).m_dev;
            }
        };

        //! The BufOmp5 dimension getter trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DimType<BufOmp5<TElem, TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The BufOmp5 memory element type get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct ElemType<BufOmp5<TElem, TDim, TIdx>>
        {
            using type = TElem;
        };

        //! The BufOmp5 extent get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetExtent<
            TIdxIntegralConst,
            BufOmp5<TElem, TDim, TIdx>,
            typename std::enable_if<(TDim::value > TIdxIntegralConst::value)>::type>
        {
            ALPAKA_FN_HOST static auto getExtent(BufOmp5<TElem, TDim, TIdx> const& extent) -> TIdx
            {
                return extent.extentElements()[TIdxIntegralConst::value];
            }
        };

        //! The BufOmp5 native pointer get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrNative<BufOmp5<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getPtrNative(BufOmp5<TElem, TDim, TIdx> const& buf) -> TElem const*
            {
                return (*buf).m_pMem;
            }
            ALPAKA_FN_HOST static auto getPtrNative(BufOmp5<TElem, TDim, TIdx>& buf) -> TElem*
            {
                return (*buf).m_pMem;
            }
        };
        //! The BufOmp5 pointer on device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufOmp5<TElem, TDim, TIdx>, DevOmp5>
        {
            ALPAKA_FN_HOST static auto getPtrDev(BufOmp5<TElem, TDim, TIdx> const& buf, DevOmp5 const& dev)
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
            ALPAKA_FN_HOST static auto getPtrDev(BufOmp5<TElem, TDim, TIdx>& buf, DevOmp5 const& dev) -> TElem*
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

        //! The BufOmp5 1D memory allocation trait specialization.
        template<typename TElem, typename TIdx>
        struct BufAlloc<TElem, DimInt<1u>, TIdx, DevOmp5>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevOmp5 const& dev, TExtent const& extent)
                -> BufOmp5<TElem, DimInt<1u>, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                auto const width(getWidth(extent));
                auto const widthBytes(width * static_cast<TIdx>(sizeof(TElem)));

                void* memPtr = omp_target_alloc(static_cast<std::size_t>(widthBytes), dev.getNativeHandle());

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << " ptr: " << memPtr
                          << " device: " << dev.getNativeHandle() << std::endl;
#    endif
                return BufOmp5<TElem, DimInt<1u>, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), extent);
            }
        };

        //! The BufOmp5 nD memory allocation trait specialization. \todo Add pitch
        template<typename TElem, typename TDim, typename TIdx>
        struct BufAlloc<TElem, TDim, TIdx, DevOmp5>
        {
            template<typename TExtent>
            ALPAKA_FN_HOST static auto allocBuf(DevOmp5 const& dev, TExtent const& extent)
                -> BufOmp5<TElem, TDim, TIdx>
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                const std::size_t size = static_cast<std::size_t>(getExtentVec(extent).prod()) * sizeof(TElem);

                void* memPtr = omp_target_alloc(size, dev.getNativeHandle());
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                std::cout << __func__ << " dim: " << TDim::value << " extent: " << getExtentVec(extent)
                          << " ewb: " << size << " ptr: " << memPtr << " device: " << dev.getNativeHandle()
                          << std::endl;
#    endif
                return BufOmp5<TElem, TDim, TIdx>(dev, reinterpret_cast<TElem*>(memPtr), extent);
            }
        };

        //! The BufOmp5 offset get trait specialization.
        template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
        struct GetOffset<TIdxIntegralConst, BufOmp5<TElem, TDim, TIdx>>
        {
            ALPAKA_FN_HOST static auto getOffset(BufOmp5<TElem, TDim, TIdx> const&) -> TIdx
            {
                return 0u;
            }
        };

        //! The BufOmp5 idx type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct IdxType<BufOmp5<TElem, TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The BufCpu pointer on OpenMP 5 device get trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevOmp5>
        {
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const&, DevOmp5 const&) -> TElem const*
            {
                throw std::runtime_error("Mapping host memory to OMP5 device not implemented!");
            }
            ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>&, DevOmp5 const&) -> TElem*
            {
                throw std::runtime_error("Mapping host memory to OMP5 device not implemented!");
            }
        };
    } // namespace trait
} // namespace alpaka

#    include <alpaka/mem/buf/omp5/Copy.hpp>
#    include <alpaka/mem/buf/omp5/Set.hpp>

#endif
