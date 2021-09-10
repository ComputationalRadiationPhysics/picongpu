/* Copyright 2021 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/boundary/Utility.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <vector>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Get a vector of all axis-aligned exchanges
            HINLINE std::vector<uint32_t> getAllAxisAlignedExchanges()
            {
                auto const numExchanges = NumberOfExchanges<simDim>::value;
                auto allExchanges = std::vector<uint32_t>(numExchanges);
                std::iota(allExchanges.begin(), allExchanges.end(), 1);
                auto result = std::vector<uint32_t>{};
                std::copy_if(
                    allExchanges.begin(),
                    allExchanges.end(),
                    std::back_inserter(result),
                    pmacc::boundary::isAxisAligned);
                return result;
            }

            /** Get boundary offset in cells for the given species and exchange
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param exchangeType exchange describing the active boundary
             */
            template<typename T_Species>
            HINLINE uint32_t getOffsetCells(T_Species& species, uint32_t exchangeType)
            {
                uint32_t axis = pmacc::boundary::getAxis(exchangeType);
                return species.boundaryDescription()[axis].offset;
            }

            /** Get a range of cells that are internal for the given species wrt given exchangeType
             *
             * Note that it only considers one boundary, the returned cells may be external for other boundaries.
             * The results are in the total coordinate system.
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param exchangeType exchange describing the active boundary
             * @param[out] begin begin of the range, all cells such that begin <= cell < end component-wise fit
             * @param[out] end end of the range, all cells such that begin <= cell < end component-wise fit
             */
            template<typename T_Species>
            HINLINE void getInternalCellsTotal(
                T_Species& species,
                uint32_t exchangeType,
                pmacc::DataSpace<simDim>* begin,
                pmacc::DataSpace<simDim>* end)
            {
                auto axis = pmacc::boundary::getAxis(exchangeType);
                auto offsetCells = getOffsetCells(species, exchangeType);
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                *begin = subGrid.getGlobalDomain().offset;
                *end = (*begin) + subGrid.getGlobalDomain().size;
                (*begin)[axis] += offsetCells;
                (*end)[axis] -= offsetCells;
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
