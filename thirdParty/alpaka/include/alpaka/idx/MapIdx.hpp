/**
* \file
* Copyright 2014-2017 Benjamin Worpitz, Axel Huebl
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

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Common.hpp>

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

namespace alpaka
{
    namespace idx
    {
        namespace detail
        {
            //#############################################################################
            //! Maps a linear index to a N dimensional index.
            template<
                std::size_t TidxDimOut,
                std::size_t TidxDimIn,
                typename TSfinae = void>
            struct MapIdx;
            //#############################################################################
            //! Maps a N dimensional index to the same N dimensional index.
            template<
                std::size_t TidxDim>
            struct MapIdx<
                TidxDim,
                TidxDim>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A N dimensional vector.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<TidxDim>, TElem> const & idx,
#if !BOOST_ARCH_CUDA_DEVICE
                    vec::Vec<dim::DimInt<TidxDim>, TElem> const & extent)
#else
                    vec::Vec<dim::DimInt<TidxDim>, TElem> const &)
#endif
                -> vec::Vec<dim::DimInt<TidxDim>, TElem>
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    boost::ignore_unused(extent);
#endif
                    return idx;
                }
            };
            //#############################################################################
            //! Maps a 1 dimensional index to a N dimensional index.
            template<
                std::size_t TidxDimOut>
            struct MapIdx<
                TidxDimOut,
                1u,
                typename std::enable_if<TidxDimOut != 1u>::type>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to
                // \return A N dimensional vector.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<1u>, TElem> const & idx,
                    vec::Vec<dim::DimInt<TidxDimOut>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<TidxDimOut>, TElem>
                {
                    auto idxNd(vec::Vec<dim::DimInt<TidxDimOut>, TElem>::all(0u));

                    constexpr std::size_t lastIdx(TidxDimOut - 1u);

                    // fast-dim
                    idxNd[lastIdx] = static_cast<TElem>(idx[0u] % extent[lastIdx]);

                    // in-between
                    TElem hyperPlanesBefore = extent[lastIdx];
                    for(std::size_t r(1u); r < lastIdx; ++r)
                    {
                        std::size_t const d = lastIdx - r;
                        idxNd[d] = static_cast<TElem>(idx[0u] / hyperPlanesBefore % extent[d]);
                        hyperPlanesBefore *= extent[d];
                    }

                    // slow-dim
                    idxNd[0u] = static_cast<TElem>(idx[0u] / hyperPlanesBefore);

                    return idxNd;
                }
            };
            //#############################################################################
            //! Maps a N dimensional index to a 1 dimensional index.
            template<
                std::size_t TidxDimIn>
            struct MapIdx<
                1u,
                TidxDimIn,
                typename std::enable_if<TidxDimIn != 1u>::type>
            {
                //-----------------------------------------------------------------------------
                // \tparam TElem Type of the index values.
                // \param idx Idx to be mapped.
                // \param extent Spatial size to map the index to.
                // \return A 1 dimensional vector.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TElem>
                ALPAKA_FN_HOST_ACC static auto mapIdx(
                    vec::Vec<dim::DimInt<TidxDimIn>, TElem> const & idx,
                    vec::Vec<dim::DimInt<TidxDimIn>, TElem> const & extent)
                -> vec::Vec<dim::DimInt<1u>, TElem>
                {
                    TElem idx1d(idx[0u]);
                    for(std::size_t d(1u); d < TidxDimIn; ++d)
                    {
                        idx1d = static_cast<TElem>(idx1d * extent[d] + idx[d]);
                    }
                    return {idx1d};
                }
            };
        }

        //#############################################################################
        //! Maps a N dimensional index to a N dimensional position.
        //!
        //! \tparam TidxDimOut Dimension of the index vector to map to.
        //! \tparam TidxDimIn Dimension of the index vector to map from.
        //! \tparam TElem Type of the elements of the index vector to map from.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            std::size_t TidxDimOut,
            std::size_t TidxDimIn,
            typename TElem>
        ALPAKA_FN_HOST_ACC auto mapIdx(
            vec::Vec<dim::DimInt<TidxDimIn>, TElem> const & idx,
            vec::Vec<dim::DimInt<(TidxDimOut < TidxDimIn) ? TidxDimIn : TidxDimOut>, TElem> const & extent)
        -> vec::Vec<dim::DimInt<TidxDimOut>, TElem>
        {
            static_assert(TidxDimOut > 0u, "The dimension of the output vector has to be greater than zero!");
            static_assert(TidxDimIn > 0u, "The dimension of the input vector has to be greater than zero!");

            return
                detail::MapIdx<
                    TidxDimOut,
                    TidxDimIn>
                ::mapIdx(
                    idx,
                    extent);
        }
    }
}
