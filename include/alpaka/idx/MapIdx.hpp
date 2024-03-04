/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Erik Zenker, Jan Stephan, Jeffrey Kelling, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/vec/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <type_traits>

namespace alpaka
{
    //! Maps an N-dimensional index to an N-dimensional position. At least one dimension must always be 1 or zero.
    //!
    //! \tparam TDimOut Dimension of the index vector to map to.
    //! \param in The index vector to map from.
    //! \param extent The extents of the input or output space, whichever has more than 1 dimensions.
    ALPAKA_NO_HOST_ACC_WARNING template<
        std::size_t TDimOut,
        std::size_t TDimIn,
        std::size_t TDimExtents,
        typename TElem>
    ALPAKA_FN_HOST_ACC auto mapIdx(Vec<DimInt<TDimIn>, TElem> const& in, Vec<DimInt<TDimExtents>, TElem> const& extent)
        -> Vec<DimInt<TDimOut>, TElem>
    {
        if constexpr(TDimOut == 0 || TDimIn == 0)
            return Vec<DimInt<TDimOut>, TElem>::zeros();
        else if constexpr(TDimOut == TDimIn)
            return in;
        else if constexpr(TDimOut == 1)
        {
            TElem out = in[0];
            for(std::size_t d = 1; d < TDimIn; ++d)
                out = static_cast<TElem>(out * extent[d] + in[d]);
            return {out};
        }
        else if constexpr(TDimIn == 1)
        {
            auto flat = in.front();
            Vec<DimInt<TDimOut>, TElem> out;
            for(std::size_t d = TDimOut - 1u; d > 0; d--)
            {
                out[d] = static_cast<TElem>(flat % extent[d]);
                flat /= extent[d];
            }
            out.front() = static_cast<TElem>(flat);
            return out;
        }
        else
            static_assert(!sizeof(TElem), "Not implemented");

        ALPAKA_UNREACHABLE({});
    }

    //! Maps an N dimensional index to a N dimensional position based on the pitches of a view without padding or a
    //! byte view. At least one dimension must always be 1 or zero.
    //!
    //! \tparam TDimOut Dimension of the index vector to map to.
    //! \param in The index vector to map from.
    //! \param pitches The pitches of the input or output space, whichever has more than 1 dimensions.
    ALPAKA_NO_HOST_ACC_WARNING
    template<std::size_t TDimOut, std::size_t TDimIn, std::size_t TidxDimPitch, typename TElem>
    ALPAKA_FN_HOST_ACC auto mapIdxPitchBytes(
        Vec<DimInt<TDimIn>, TElem> const& in,
        Vec<DimInt<TidxDimPitch>, TElem> const& pitches) -> Vec<DimInt<TDimOut>, TElem>
    {
        if constexpr(TDimOut == 0 || TDimIn == 0)
            return Vec<DimInt<TDimOut>, TElem>::zeros();
        else if constexpr(TDimOut == TDimIn)
            return in;
        else if constexpr(TDimOut == 1)
        {
            using DimMinusOne = DimInt<TDimIn - 1>;
            return {in.back() + (subVecBegin<DimMinusOne>(pitches) * subVecBegin<DimMinusOne>(in)).sum()};
        }
        else if constexpr(TDimIn == 1)
        {
            auto result = Vec<DimInt<TDimOut>, TElem>::zeros();

            TElem out = in.front();
            for(std::size_t d = 0; d < TDimOut - 1u; ++d)
            {
                result[d] = static_cast<TElem>(out / pitches[d]);
                out %= pitches[d];
            }
            result.back() = out;

            return result;
        }
        else
            static_assert(!sizeof(TElem), "Not implemented");

        ALPAKA_UNREACHABLE({});
    }
} // namespace alpaka
