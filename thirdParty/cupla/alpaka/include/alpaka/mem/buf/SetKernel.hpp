/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/idx/Accessors.hpp"
#include "alpaka/idx/MapIdx.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/meta/Fold.hpp"

namespace alpaka
{
    //! any device ND memory set kernel.
    class MemSetKernel
    {
    public:
        //! The kernel entry point.
        //!
        //! All but the last element of threadElemExtent must be one.
        //!
        //! \tparam TAcc The accelerator environment to be executed on.
        //! \tparam TExtent extent type.
        //! \param acc The accelerator to be executed on.
        //! \param val value to set.
        //! \param dst target mem ptr.
        //! \param extent area to set.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TAcc, typename TExtent, typename TPitch>
        ALPAKA_FN_ACC auto operator()(
            TAcc const& acc,
            std::uint8_t const val,
            std::uint8_t* dst,
            TExtent extent,
            TPitch pitch) const -> void
        {
            using Idx = typename alpaka::trait::IdxType<TExtent>::type;
            auto const gridThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            auto const threadElemExtent(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
            auto const idxThreadFirstElem = getIdxThreadFirstElem(acc, gridThreadIdx, threadElemExtent);
            auto idx = mapIdxPitchBytes<1u, Dim<TAcc>::value>(idxThreadFirstElem, pitch)[0];
            constexpr auto lastDim = Dim<TAcc>::value - 1;
            auto const lastIdx = idx
                                 + std::min(
                                     threadElemExtent[lastDim],
                                     static_cast<Idx>(extent[lastDim] - idxThreadFirstElem[lastDim]));

            if((idxThreadFirstElem < extent).foldrAll(std::logical_and<bool>()))
            {
                for(; idx < lastIdx; ++idx)
                {
                    *(dst + idx) = val;
                }
            }
        }
    };
} // namespace alpaka
