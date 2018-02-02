/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/acc/Traits.hpp>

#include <alpaka/vec/Vec.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Common.hpp>

#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>
#include <set>
#include <array>

//-----------------------------------------------------------------------------
//! The alpaka library.
namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The grid block extent subdivision restrictions.
        enum class GridBlockExtentSubDivRestrictions
        {
            EqualExtent,       //!< The block thread extent will be equal in all dimensions.
            CloseToEqualExtent,//!< The block thread extent will be as close to equal as possible in all dimensions.
            Unrestricted,      //!< The block thread extent will not have any restrictions.
        };

        namespace detail
        {
            //-----------------------------------------------------------------------------
            //! \param maxDivisor The maximum divisor.
            //! \param dividend The dividend.
            //! \return The biggest number that satisfies the following conditions:
            //!     1) dividend/ret==0
            //!     2) ret<=maxDivisor
            template<
                typename T,
                typename = typename std::enable_if<std::is_integral<T>::value>::type>
            ALPAKA_FN_HOST auto nextDivisorLowerOrEqual(
                T const & maxDivisor,
                T const & dividend)
            -> T
            {
                T divisor(maxDivisor);

                core::assertValueUnsigned(dividend);
                core::assertValueUnsigned(maxDivisor);
                assert(dividend <= maxDivisor);

                while((dividend%divisor) != 0)
                {
                    --divisor;
                }

                return divisor;
            }
            //-----------------------------------------------------------------------------
            //! \param val The value to find divisors of.
            //! \param maxDivisor The maximum.
            //! \return A list of all divisors less then or equal to the given maximum.
            template<
                typename T,
                typename = typename std::enable_if<std::is_integral<T>::value>::type>
            ALPAKA_FN_HOST auto allDivisorsLessOrEqual(
                T const & val,
                T const & maxDivisor)
            -> std::set<T>
            {
                std::set<T> divisorSet;

                core::assertValueUnsigned(val);
                core::assertValueUnsigned(maxDivisor);
                assert(maxDivisor <= val);

                for(T i(1); i <= std::min(val, maxDivisor); ++i)
                {
                    if(val % i == 0)
                    {
                        divisorSet.insert(val/i);
                    }
                }

                return divisorSet;
            }
        }

        //-----------------------------------------------------------------------------
        //! \tparam TDim The dimensionality of the accelerator device properties.
        //! \tparam TSize The size type of the accelerator device properties.
        //! \param accDevProps The maxima for the work division.
        //! \return If the accelerator device properties are valid.
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto isValidAccDevProps(
            acc::AccDevProps<TDim, TSize> const & accDevProps)
        -> bool
        {
            // Check that the maximum counts are greater or equal 1.
            if((accDevProps.m_gridBlockCountMax < 1)
                || (accDevProps.m_blockThreadCountMax < 1)
                || (accDevProps.m_threadElemCountMax < 1))
            {
                return false;
            }

            // Store the maxima allowed for extents of grid, blocks and threads.
            auto const gridBlockExtentMax(vec::subVecEnd<TDim>(accDevProps.m_gridBlockExtentMax));
            auto const blockThreadExtentMax(vec::subVecEnd<TDim>(accDevProps.m_blockThreadExtentMax));
            auto const threadElemExtentMax(vec::subVecEnd<TDim>(accDevProps.m_threadElemExtentMax));

            // Check that the extents for all dimensions are correct.
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                // Check that the maximum extents are greater or equal 1.
                if((gridBlockExtentMax[i] < 1)
                    || (blockThreadExtentMax[i] < 1)
                    || (threadElemExtentMax[i] < 1))
                {
                    return false;
                }
            }

            return true;
        }

        //-----------------------------------------------------------------------------
        //! Subdivides the given grid thread extent into blocks restricted by the maxima allowed.
        //! 1. The the maxima block, thread and element extent and counts
        //! 2. The requirement of the block thread extent to divide the grid thread extent without remainder
        //! 3. The requirement of the block extent.
        //!
        //! \param gridElemExtent
        //!     The full extent of elements in the grid.
        //! \param threadElemExtent
        //!     the number of elements computed per thread.
        //! \param accDevProps
        //!     The maxima for the work division.
        //! \param requireBlockThreadExtentToDivideGridThreadExtent
        //!     If this is true, the grid thread extent will be multiples of the corresponding block thread extent.
        //!     NOTE: If this is true and gridThreadExtent is prime (or otherwise bad chosen) in a dimension, the block thread extent will be one in this dimension.
        //! \param gridBlockExtentSubDivRestrictions
        //!     The grid block extent subdivision restrictions.
        template<
            typename TDim,
            typename TSize>
        ALPAKA_FN_HOST auto subDivideGridElems(
            vec::Vec<TDim, TSize> const & gridElemExtent,
            vec::Vec<TDim, TSize> threadElemExtent,
            acc::AccDevProps<TDim, TSize> const & accDevProps,
            bool requireBlockThreadExtentToDivideGridThreadExtent = true,
            GridBlockExtentSubDivRestrictions gridBlockExtentSubDivRestrictions = GridBlockExtentSubDivRestrictions::Unrestricted)
        -> workdiv::WorkDivMembers<TDim, TSize>
        {
            ///////////////////////////////////////////////////////////////////
            // Check that the input data is valid.
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                assert(gridElemExtent[i] >= 1);
                assert(threadElemExtent[i] >= 1);
                assert(threadElemExtent[i] <= accDevProps.m_threadElemExtentMax[i]);
            }
            assert(threadElemExtent.prod() <= accDevProps.m_threadElemCountMax);
            assert(isValidAccDevProps(accDevProps));

            ///////////////////////////////////////////////////////////////////
            // Handle the given threadElemExtent. After this only the blockThreadExtent has to be optimized.

            // Restrict the thread elem extent with the grid elem extent.
            for(typename TDim::value_type i(0); i<TDim::value; ++i)
            {
                threadElemExtent[i] = std::min(threadElemExtent[i], gridElemExtent[i]);
            }

            // Calculate the grid thread extent.
            auto gridThreadExtent(vec::Vec<TDim, TSize>::zeros());
            for(typename TDim::value_type i(0u); i<TDim::value; ++i)
            {
                gridThreadExtent[i] =
                    static_cast<TSize>(
                        std::ceil(
                            static_cast<double>(gridElemExtent[i])
                            / static_cast<double>(threadElemExtent[i])));
            }

            ///////////////////////////////////////////////////////////////////
            // Try to calculate an optimal blockThreadExtent.

            // Initialize the block thread extent with the maximum possible.
            auto blockThreadExtent(accDevProps.m_blockThreadExtentMax);

            // Restrict the max block thread extent with the grid thread extent.
            // This removes dimensions not required in the grid thread extent.
            // This has to be done before the blockThreadCountMax clipping to get the maximum correctly.
            for(typename TDim::value_type i(0u); i<TDim::value; ++i)
            {
                blockThreadExtent[i] = std::min(blockThreadExtent[i], gridThreadExtent[i]);
            }

            // For equal block thread extent, restrict it to its minimum component.
            // For example (512, 256, 1024) will get (256, 256, 256).
            if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::EqualExtent)
            {
                auto const minBlockThreadExtent(blockThreadExtent.min());
                for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                {
                    blockThreadExtent[i] = minBlockThreadExtent;
                }
            }

            auto const & blockThreadCountMax(accDevProps.m_blockThreadCountMax);
            // Adjust blockThreadExtent if its product is too large.
            if(blockThreadExtent.prod() > blockThreadCountMax)
            {
                // Satisfy the following equation:
                // blockThreadCountMax >= blockThreadExtent.prod()
                // For example 1024 >= 512 * 512 * 1024

                // For equal block thread extent this is easily the nth root of blockThreadCountMax.
                if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::EqualExtent)
                {
                    double const fNthRoot(std::pow(blockThreadCountMax, 1.0/static_cast<double>(TDim::value)));
                    TSize const nthRoot(static_cast<TSize>(fNthRoot));
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        blockThreadExtent[i] = nthRoot;
                    }
                }
                else if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::CloseToEqualExtent)
                {
                    // Very primitive clipping. Just halve the largest value until it fits.
                    while(blockThreadExtent.prod() > blockThreadCountMax)
                    {
                        auto const maxElemIdx(blockThreadExtent.maxElem());
                        blockThreadExtent[maxElemIdx] = blockThreadExtent[maxElemIdx] / static_cast<TSize>(2u);
                    }
                }
                else
                {
                    // Very primitive clipping. Just halve the smallest value until it fits.
                    while(blockThreadExtent.prod() > blockThreadCountMax)
                    {
                        // Compute the minimum element index but ignore ones.
                        // Ones compare always larger to everything else.
                        auto const minElemIdx(
                            static_cast<TSize>(
                                std::distance(
                                    &blockThreadExtent[0u],
                                    std::min_element(
                                        &blockThreadExtent[0u],
                                        &blockThreadExtent[TDim::value-1u],
                                        [](TSize const & a, TSize const & b)
                                        {
                                            // This first case is redundant.
                                            /*if((a == 1u) && (b == 1u))
                                            {
                                                return false;
                                            }
                                            else */if(a == static_cast<TSize>(1u))
                                            {
                                                return false;
                                            }
                                            else if(b == static_cast<TSize>(1u))
                                            {
                                                return true;
                                            }
                                            else
                                            {
                                                return a < b;
                                            }
                                        }))));
                        blockThreadExtent[minElemIdx] = blockThreadExtent[minElemIdx] / static_cast<TSize>(2u);
                    }
                }
            }

            // Make the block thread extent divide the grid thread extent.
            if(requireBlockThreadExtentToDivideGridThreadExtent)
            {
                if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::EqualExtent)
                {
                    // For equal size block extent we have to compute the gcd of all grid thread extent that is less then the current maximal block thread extent.
                    // For this we compute the divisors of all grid thread extent less then the current maximal block thread extent.
                    std::array<std::set<TSize>, TDim::value> gridThreadExtentDivisors;
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        gridThreadExtentDivisors[i] =
                            detail::allDivisorsLessOrEqual(
                                gridThreadExtent[i],
                                blockThreadExtent[i]);
                    }
                    // The maximal common divisor of all block thread extent is the optimal solution.
                    std::set<TSize> intersects[2u];
                    for(typename TDim::value_type i(1u); i<TDim::value; ++i)
                    {
                        intersects[(i-1u)%2u] = gridThreadExtentDivisors[0];
                        intersects[(i)%2u].clear();
                        set_intersection(
                            intersects[(i-1u)%2u].begin(),
                            intersects[(i-1u)%2u].end(),
                            gridThreadExtentDivisors[i].begin(),
                            gridThreadExtentDivisors[i].end(),
                            std::inserter(intersects[i%2], intersects[i%2u].begin()));
                    }
                    TSize const maxCommonDivisor(*(--intersects[(TDim::value-1)%2u].end()));
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        blockThreadExtent[i] = maxCommonDivisor;
                    }
                }
                else if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::CloseToEqualExtent)
                {
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        blockThreadExtent[i] =
                            detail::nextDivisorLowerOrEqual(
                                blockThreadExtent[i],
                                gridThreadExtent[i]);
                    }
                }
                else
                {
                    for(typename TDim::value_type i(0u); i<TDim::value; ++i)
                    {
                        blockThreadExtent[i] =
                            detail::nextDivisorLowerOrEqual(
                                blockThreadExtent[i],
                                gridThreadExtent[i]);
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////
            // Compute the gridBlockExtent.

            // Set the grid block extent (rounded to the next integer not less then the quotient.
            auto gridBlockExtent(vec::Vec<TDim, TSize>::ones());
            for(typename TDim::value_type i(0u); i<TDim::value; ++i)
            {
                gridBlockExtent[i] =
                    static_cast<TSize>(
                        std::ceil(
                            static_cast<double>(gridThreadExtent[i])
                            / static_cast<double>(blockThreadExtent[i])));
            }

            ///////////////////////////////////////////////////////////////////
            // Return the final work division.
            return
                workdiv::WorkDivMembers<TDim, TSize>(
                    gridBlockExtent,
                    blockThreadExtent,
                    threadElemExtent);
        }

        //-----------------------------------------------------------------------------
        //! \tparam TAcc The accelerator for which this work division has to be valid.
        //! \tparam TGridElemExtent The type of the grid element extent.
        //! \tparam TThreadElemExtent The type of the thread element extent.
        //! \tparam TDev The type of the device.
        //! \param dev
        //!     The device the work division should be valid for.
        //! \param gridElemExtent
        //!     The full extent of elements in the grid.
        //! \param threadElemExtents
        //!     the number of elements computed per thread.
        //! \param requireBlockThreadExtentToDivideGridThreadExtent
        //!     If this is true, the grid thread extent will be multiples of the corresponding block thread extent.
        //!     NOTE: If this is true and gridThreadExtent is prime (or otherwise bad chosen) in a dimension, the block thread extent will be one in this dimension.
        //! \param gridBlockExtentSubDivRestrictions
        //!     The grid block extent subdivision restrictions.
        //! \return The work division.
        template<
            typename TAcc,
            typename TGridElemExtent,
            typename TThreadElemExtent,
            typename TDev>
        ALPAKA_FN_HOST auto getValidWorkDiv(
            TDev const & dev,
            TGridElemExtent const & gridElemExtent = TGridElemExtent(),
            TThreadElemExtent const & threadElemExtents = TThreadElemExtent(),
            bool requireBlockThreadExtentToDivideGridThreadExtent = true,
            GridBlockExtentSubDivRestrictions gridBlockExtentSubDivRestrictions = GridBlockExtentSubDivRestrictions::Unrestricted)
        -> workdiv::WorkDivMembers<dim::Dim<TGridElemExtent>, size::Size<TGridElemExtent>>
        {
            static_assert(
                dim::Dim<TGridElemExtent>::value == dim::Dim<TAcc>::value,
                "The dimension of TAcc and the dimension of TGridElemExtent have to be identical!");
            static_assert(
                dim::Dim<TThreadElemExtent>::value == dim::Dim<TAcc>::value,
                "The dimension of TAcc and the dimension of TThreadElemExtent have to be identical!");
            static_assert(
                std::is_same<size::Size<TGridElemExtent>, size::Size<TAcc>>::value,
                "The size type of TAcc and the size type of TGridElemExtent have to be identical!");
            static_assert(
                std::is_same<size::Size<TThreadElemExtent>, size::Size<TAcc>>::value,
                "The size type of TAcc and the size type of TThreadElemExtent have to be identical!");

            return subDivideGridElems(
                extent::getExtentVec(gridElemExtent),
                extent::getExtentVec(threadElemExtents),
                acc::getAccDevProps<TAcc>(dev),
                requireBlockThreadExtentToDivideGridThreadExtent,
                gridBlockExtentSubDivRestrictions);
        }

        //-----------------------------------------------------------------------------
        //! \tparam TDim The dimensionality of the accelerator device properties.
        //! \tparam TSize The size type of the accelerator device properties.
        //! \tparam TWorkDiv The type of the work division.
        //! \param accDevProps The maxima for the work division.
        //! \param workDiv The work division to test for validity.
        //! \return If the work division is valid for the given accelerator device properties.
        template<
            typename TDim,
            typename TSize,
            typename TWorkDiv>
        ALPAKA_FN_HOST auto isValidWorkDiv(
            acc::AccDevProps<TDim, TSize> const & accDevProps,
            TWorkDiv const & workDiv)
        -> bool
        {
            // Store the maxima allowed for extents of grid, blocks and threads.
            auto const gridBlockExtentMax(vec::subVecEnd<dim::Dim<TWorkDiv>>(accDevProps.m_gridBlockExtentMax));
            auto const blockThreadExtentMax(vec::subVecEnd<dim::Dim<TWorkDiv>>(accDevProps.m_blockThreadExtentMax));
            auto const threadElemExtentMax(vec::subVecEnd<dim::Dim<TWorkDiv>>(accDevProps.m_threadElemExtentMax));

            // Get the extents of grid, blocks and threads of the work division to check.
            auto const gridBlockExtent(getWorkDiv<Grid, Blocks>(workDiv));
            auto const blockThreadExtent(getWorkDiv<Block, Threads>(workDiv));
            auto const threadElemExtent(getWorkDiv<Block, Threads>(workDiv));

            // Check that the maximal counts are satisfied.
            if(accDevProps.m_gridBlockCountMax < gridBlockExtent.prod())
            {
                return false;
            }
            if(accDevProps.m_blockThreadCountMax < blockThreadExtent.prod())
            {
                return false;
            }
            if(accDevProps.m_threadElemCountMax < threadElemExtent.prod())
            {
                return false;
            }

            // Check that the extents for all dimensions are correct.
            for(typename dim::Dim<TWorkDiv>::value_type i(0); i<dim::Dim<TWorkDiv>::value; ++i)
            {
                // No extent is allowed to be zero or greater then the allowed maximum.
                if((gridBlockExtent[i] < 1)
                    || (blockThreadExtent[i] < 1)
                    || (threadElemExtent[i] < 1)
                    || (gridBlockExtentMax[i] < gridBlockExtent[i])
                    || (blockThreadExtentMax[i] < blockThreadExtent[i])
                    || (threadElemExtentMax[i] < threadElemExtent[i]))
                {
                    return false;
                }
            }

            return true;
        }
        //-----------------------------------------------------------------------------
        //! \tparam TAcc The accelerator to test the validity on.
        //! \param dev The device to test the work division for validity on.
        //! \param workDiv The work division to test for validity.
        //! \return If the work division is valid on this accelerator.
        template<
            typename TAcc,
            typename TDev,
            typename TWorkDiv>
        ALPAKA_FN_HOST auto isValidWorkDiv(
            TDev const & dev,
            TWorkDiv const & workDiv)
        -> bool
        {
            return
                workdiv::isValidWorkDiv(
                    acc::getAccDevProps<TAcc>(dev),
                    workDiv);
        }
    }
}
