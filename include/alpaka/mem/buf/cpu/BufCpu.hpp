/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpuImpl.hpp"
#include "alpaka/mem/buf/cpu/ConstBufCpu.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    //! The CPU memory buffer template implementing muting accessors.
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu : public internal::ViewAccessorType<DevCpu, BufCpu<TElem, TDim, TIdx>>
    {
        using TBufImpl = detail::BufCpuImpl<TElem, TDim, TIdx>;

    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST BufCpu(DevCpu const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_spBufImpl{std::make_shared<TBufImpl>(dev, pMem, std::move(deleter), extent)}
        {
        }

    public:
        std::shared_ptr<TBufImpl> m_spBufImpl;
    };
} // namespace alpaka

#include "alpaka/mem/buf/cpu/Copy.hpp"
#include "alpaka/mem/buf/cpu/Fill.hpp"
#include "alpaka/mem/buf/cpu/Set.hpp"
#include "alpaka/mem/buf/cpu/traits/BufCpuTraits.hpp"
#include "alpaka/mem/buf/cpu/traits/ConstBufCpuTraits.hpp"
