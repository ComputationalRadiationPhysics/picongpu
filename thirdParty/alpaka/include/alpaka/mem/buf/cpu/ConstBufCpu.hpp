/* Copyright 2022 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Andrea Bocci, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/core/ApiHipRt.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/core/Vectorize.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/mem/alloc/AllocCpuAligned.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/buf/cpu/BufCpuImpl.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/meta/DependentFalseType.hpp"
#include "alpaka/platform/PlatformCpu.hpp"
#include "alpaka/vec/Vec.hpp"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
    // Predeclaration of BufCpu
    template<typename TElem, typename TDim, typename TIdx>
    class BufCpu;

    //! The CPU memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class ConstBufCpu : public internal::ViewAccessorType<DevCpu, ConstBufCpu<TElem, TDim, TIdx>>
    {
    public:
        template<typename TExtent, typename Deleter>
        ALPAKA_FN_HOST ConstBufCpu(DevCpu const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
            : m_spBufImpl{
                std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
        {
        }

        ALPAKA_FN_HOST ConstBufCpu(BufCpu<TElem, TDim, TIdx> const& buf) : m_spBufImpl{buf.m_spBufImpl}
        {
        }

        ALPAKA_FN_HOST ConstBufCpu(BufCpu<TElem, TDim, TIdx>&& buf) : m_spBufImpl{std::move(buf.m_spBufImpl)}
        {
        }

    private:
        std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufImpl;

        friend alpaka::trait::GetDev<ConstBufCpu<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetExtents<ConstBufCpu<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrNative<ConstBufCpu<TElem, TDim, TIdx>>;
        friend alpaka::trait::GetPtrDev<ConstBufCpu<TElem, TDim, TIdx>, DevCpu>;
    };

} // namespace alpaka
