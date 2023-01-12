/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Erik Zenker, Jan Stephan, Jeffrey Kelling, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/vec/Vec.hpp>

#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        //! Maps a linear index to a N dimensional index.
        template<std::size_t TidxDimOut, std::size_t TidxDimIn, typename TSfinae = void>
        struct MapIdx;
        //! Maps a N dimensional index to the same N dimensional index.
        template<std::size_t TidxDim>
        struct MapIdx<TidxDim, TidxDim>
        {
            // \tparam TElem Type of the index values.
            // \param idx Idx to be mapped.
            // \param extent Spatial size to map the index to.
            // \return A N dimensional vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<DimInt<TidxDim>, TElem> const& idx,
                [[maybe_unused]] Vec<DimInt<TidxDim>, TElem> const& extent) -> Vec<DimInt<TidxDim>, TElem>
            {
                return idx;
            }
        };
        //! Maps a 1 dimensional index to a N dimensional index.
        template<std::size_t TidxDimOut>
        struct MapIdx<TidxDimOut, 1u, std::enable_if_t<(TidxDimOut > 1u)>>
        {
            // \tparam TElem Type of the index values.
            // \param idx Idx to be mapped.
            // \param extent Spatial size to map the index to
            // \return A N dimensional vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<DimInt<1u>, TElem> const& idx,
                Vec<DimInt<TidxDimOut>, TElem> const& extent) -> Vec<DimInt<TidxDimOut>, TElem>
            {
                auto idxNd = Vec<DimInt<TidxDimOut>, TElem>::all(0u);

                constexpr std::size_t lastIdx(TidxDimOut - 1u);

                // fast-dim
                idxNd[lastIdx] = static_cast<TElem>(idx[0u] % extent[lastIdx]);

                // in-between
                TElem hyperPlanesBefore = extent[lastIdx];
#if BOOST_COMP_PGI && (defined(ALPAKA_ACC_ANY_BT_OMP5_ENABLED) || defined(ALPAKA_ACC_ANY_BT_OACC_ENABLED))            \
    && !defined(TPR30645)
                for(std::size_t a(0u); a < lastIdx - 1; ++a)
                {
                    auto const r = a + 1; // NVHPC does sometimes not understand, that loops which do not start at zero
                                          // can have zero iterations
#else
                for(std::size_t r(1u); r < lastIdx; ++r)
                {
#endif
                    std::size_t const d = lastIdx - r;
                    idxNd[d] = static_cast<TElem>(idx[0u] / hyperPlanesBefore % extent[d]);
                    hyperPlanesBefore *= extent[d];
                }

                // slow-dim
                idxNd[0u] = static_cast<TElem>(idx[0u] / hyperPlanesBefore);

                return idxNd;
            }
        };
        //! Maps a 1 dimensional index to a N dimensional index.
        template<std::size_t TidxDimIn>
        struct MapIdx<1u, TidxDimIn, std::enable_if_t<(TidxDimIn > 1u)>>
        {
            // \tparam TElem Type of the index values.
            // \param idx Idx to be mapped.
            // \param extent Spatial size to map the index to.
            // \return A 1 dimensional vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<DimInt<TidxDimIn>, TElem> const& idx,
                Vec<DimInt<TidxDimIn>, TElem> const& extent) -> Vec<DimInt<1u>, TElem>
            {
                TElem idx1d(idx[0u]);
                for(std::size_t d(1u); d < TidxDimIn; ++d)
                {
                    idx1d = static_cast<TElem>(idx1d * extent[d] + idx[d]);
                }
                return {idx1d};
            }
        };

        template<std::size_t TidxDimOut>
        struct MapIdx<TidxDimOut, 0u>
        {
            template<typename TElem, std::size_t TidxDimExtents>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<DimInt<0u>, TElem> const&,
                Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<TidxDimOut>, TElem>
            {
                return Vec<DimInt<TidxDimOut>, TElem>::zeros();
            }
        };

        template<std::size_t TidxDimIn>
        struct MapIdx<0u, TidxDimIn>
        {
            template<typename TElem, std::size_t TidxDimExtents>
            ALPAKA_FN_HOST_ACC static auto mapIdx(
                Vec<DimInt<TidxDimIn>, TElem> const&,
                Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<0u>, TElem>
            {
                return {};
            }
        };
    } // namespace detail

    //! Maps a N dimensional index to a N dimensional position.
    //!
    //! \tparam TidxDimOut Dimension of the index vector to map to.
    //! \tparam TidxDimIn Dimension of the index vector to map from.
    //! \tparam TElem Type of the elements of the index vector to map from.
    ALPAKA_NO_HOST_ACC_WARNING template<
        std::size_t TidxDimOut,
        std::size_t TidxDimIn,
        std::size_t TidxDimExtents,
        typename TElem>
    ALPAKA_FN_HOST_ACC auto mapIdx(
        Vec<DimInt<TidxDimIn>, TElem> const& idx,
        Vec<DimInt<TidxDimExtents>, TElem> const& extent) -> Vec<DimInt<TidxDimOut>, TElem>
    {
        return detail::MapIdx<TidxDimOut, TidxDimIn>::mapIdx(idx, extent);
    }

    namespace detail
    {
        //! Maps a linear index to a N dimensional index assuming a buffer wihtout padding.
        template<std::size_t TidxDimOut, std::size_t TidxDimIn, typename TSfinae = void>
        struct MapIdxPitchBytes;
        //! Maps a N dimensional index to the same N dimensional index assuming a buffer wihtout padding.
        template<std::size_t TidxDim>
        struct MapIdxPitchBytes<TidxDim, TidxDim>
        {
            // \tparam TElem Type of the index values.
            // \param idx Idx to be mapped.
            // \param pitch Spatial pitch (in elems) to map the index to
            // \return N dimensional vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
                Vec<DimInt<TidxDim>, TElem> const& idx,
                [[maybe_unused]] Vec<DimInt<TidxDim>, TElem> const& pitch) -> Vec<DimInt<TidxDim>, TElem>
            {
                return idx;
            }
        };
        //! Maps a 1 dimensional index to a N dimensional index assuming a buffer wihtout padding.
        template<std::size_t TidxDimOut>
        struct MapIdxPitchBytes<TidxDimOut, 1u, std::enable_if_t<(TidxDimOut > 1u)>>
        {
            // \tparam TElem Type of the index values.
            // \param idx Idx to be mapped.
            // \param pitch Spatial pitch (in elems) to map the index to
            // \return N dimensional vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
                Vec<DimInt<1u>, TElem> const& idx,
                Vec<DimInt<TidxDimOut>, TElem> const& pitch) -> Vec<DimInt<TidxDimOut>, TElem>
            {
                auto idxNd = Vec<DimInt<TidxDimOut>, TElem>::all(0u);

                constexpr std::size_t lastIdx = TidxDimOut - 1u;

                TElem tmp = idx[0u];
                for(std::size_t d(0u); d < lastIdx; ++d)
                {
                    idxNd[d] = static_cast<TElem>(tmp / pitch[d + 1]);
                    tmp %= pitch[d + 1];
                }
                idxNd[lastIdx] = tmp;

                return idxNd;
            }
        };
        //! Maps a N dimensional index to a 1 dimensional index assuming a buffer without padding.
        template<std::size_t TidxDimIn>
        struct MapIdxPitchBytes<1u, TidxDimIn, std::enable_if_t<(TidxDimIn > 1u)>>
        {
            // \tparam TElem Type of the index values.
            // \param idx Idx to be mapped.
            // \param pitch Spatial pitch (in elems) to map the index to
            // \return A 1 dimensional vector.
            ALPAKA_NO_HOST_ACC_WARNING
            template<typename TElem>
            ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
                Vec<DimInt<TidxDimIn>, TElem> const& idx,
                Vec<DimInt<TidxDimIn>, TElem> const& pitch) -> Vec<DimInt<1u>, TElem>
            {
                constexpr auto lastDim = TidxDimIn - 1;
                TElem idx1d = idx[lastDim];
                for(std::size_t d(0u); d < lastDim; ++d)
                {
                    idx1d = static_cast<TElem>(idx1d + pitch[d + 1] * idx[d]);
                }
                return {idx1d};
            }
        };

        template<std::size_t TidxDimOut>
        struct MapIdxPitchBytes<TidxDimOut, 0u>
        {
            template<typename TElem, std::size_t TidxDimExtents>
            ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
                Vec<DimInt<0u>, TElem> const&,
                Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<TidxDimOut>, TElem>
            {
                return Vec<DimInt<TidxDimOut>, TElem>::zeros();
            }
        };

        template<std::size_t TidxDimIn>
        struct MapIdxPitchBytes<0u, TidxDimIn>
        {
            template<typename TElem, std::size_t TidxDimExtents>
            ALPAKA_FN_HOST_ACC static auto mapIdxPitchBytes(
                Vec<DimInt<TidxDimIn>, TElem> const&,
                Vec<DimInt<TidxDimExtents>, TElem> const&) -> Vec<DimInt<0u>, TElem>
            {
                return {};
            }
        };
    } // namespace detail

    //! Maps a N dimensional index to a N dimensional position based on
    //! pitch in a buffer without padding or a byte buffer.
    //!
    //! \tparam TidxDimOut Dimension of the index vector to map to.
    //! \tparam TidxDimIn Dimension of the index vector to map from.
    //! \tparam TElem Type of the elements of the index vector to map from.
    ALPAKA_NO_HOST_ACC_WARNING
    template<std::size_t TidxDimOut, std::size_t TidxDimIn, std::size_t TidxDimPitch, typename TElem>
    ALPAKA_FN_HOST_ACC auto mapIdxPitchBytes(
        Vec<DimInt<TidxDimIn>, TElem> const& idx,
        Vec<DimInt<TidxDimPitch>, TElem> const& pitch) -> Vec<DimInt<TidxDimOut>, TElem>
    {
        return detail::MapIdxPitchBytes<TidxDimOut, TidxDimIn>::mapIdxPitchBytes(idx, pitch);
    }
} // namespace alpaka
