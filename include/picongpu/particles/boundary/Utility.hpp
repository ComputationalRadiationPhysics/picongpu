/* Copyright 2021-2023 Sergei Bastrakov
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

#include <pmacc/Environment.hpp>
#include <pmacc/boundary/Utility.hpp>
#include <pmacc/mappings/kernel/IntervalMapping.hpp>

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
                // We need the range 1 <= exchange < numExchanges, so -1 here
                auto allExchanges = std::vector<uint32_t>(numExchanges - 1);
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
            HINLINE uint32_t getOffsetCells(T_Species const& species, uint32_t exchangeType)
            {
                uint32_t axis = pmacc::boundary::getAxis(exchangeType);
                return species.boundaryDescription()[axis].offset;
            }

            /** Get boundary temperature in keV for the given species and exchange
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param exchangeType exchange describing the active boundary
             */
            template<typename T_Species>
            HINLINE float_X getTemperature(T_Species const& species, uint32_t exchangeType)
            {
                uint32_t axis = pmacc::boundary::getAxis(exchangeType);
                return species.boundaryDescription()[axis].temperature;
            }

            /** Get a range of cells that define external area for the given species wrt given exchangeType
             *
             * Note that it only considers one, given, boundary.
             * So the particles inside the returned range crossed that boundary.
             * Particles outside the returned range did not cross that boundary, but may still be outside for others.
             *
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
            HINLINE void getExternalCellsTotal(
                T_Species const& species,
                uint32_t exchangeType,
                pmacc::DataSpace<simDim>* begin,
                pmacc::DataSpace<simDim>* end)
            {
                auto axis = pmacc::boundary::getAxis(exchangeType);
                auto offsetCells = getOffsetCells(species, exchangeType);
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                // For non-axis directions, we take all cells including the guards
                auto const mappingDescription = species.getCellDescription();
                auto const guardCells
                    = mappingDescription.getGuardingSuperCells() * mappingDescription.getSuperCellSize();
                *begin = subGrid.getGlobalDomain().offset - guardCells;
                *end = (*begin) + subGrid.getGlobalDomain().size + guardCells * 2;
                if(pmacc::boundary::isMinSide(exchangeType))
                    (*end)[axis] = (*begin)[axis] + guardCells[axis] + offsetCells;
                if(pmacc::boundary::isMaxSide(exchangeType))
                    (*begin)[axis] = (*end)[axis] - (guardCells[axis] + offsetCells);
            }

            /** Get a range of cells that define internal area for the given species wrt given exchangeType
             *
             * Note that it only considers one, given, boundary.
             * So the particles outside the returned range crossed that boundary.
             * Particles in the returned range did not cross that boundary, but may still be outside for others.
             *
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
                T_Species const& species,
                uint32_t exchangeType,
                pmacc::DataSpace<simDim>* begin,
                pmacc::DataSpace<simDim>* end)
            {
                auto axis = pmacc::boundary::getAxis(exchangeType);
                auto offsetCells = getOffsetCells(species, exchangeType);
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                // For non-axis directions, we take all cells including the guards
                auto const mappingDescription = species.getCellDescription();
                auto const guardCells
                    = mappingDescription.getGuardingSuperCells() * mappingDescription.getSuperCellSize();
                *begin = subGrid.getGlobalDomain().offset - guardCells;
                *end = (*begin) + subGrid.getGlobalDomain().size + guardCells * 2;
                if(pmacc::boundary::isMinSide(exchangeType))
                    (*begin)[axis] += guardCells[axis] + offsetCells;
                if(pmacc::boundary::isMaxSide(exchangeType))
                    (*end)[axis] -= guardCells[axis] + offsetCells;
            }

            /** Get a range of cells that define internal area for the given species with respect to all boundaries
             *
             * The results are in the total coordinate system.
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param[out] begin begin of the range, all cells such that begin <= cell < end component-wise fit
             * @param[out] end end of the range, all cells such that begin <= cell < end component-wise fit
             */
            template<typename T_Species>
            HINLINE void getInternalCellsTotal(
                T_Species const& species,
                pmacc::DataSpace<simDim>* begin,
                pmacc::DataSpace<simDim>* end)
            {
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                *begin = subGrid.getGlobalDomain().offset;
                *end = (*begin) + subGrid.getGlobalDomain().size;
                auto const communicationMask = Environment<simDim>::get().GridController().getCommunicationMask();
                for(auto exchangeType : getAllAxisAlignedExchanges())
                {
                    auto axis = pmacc::boundary::getAxis(exchangeType);
                    auto offsetCells = getOffsetCells(species, exchangeType);
                    if(pmacc::boundary::isMinSide(exchangeType))
                        (*begin)[axis] += offsetCells;
                    if(pmacc::boundary::isMaxSide(exchangeType))
                        (*end)[axis] -= offsetCells;
                }
            }

            /** Get a mapper factory that define active supercells for the given species wrt given exchangeType
             *
             * The area only containes the supercells with the "just crossed" particles, not all external ones.
             * It relies on the fact paricles can't pass more than 1 cell in each direction per time step.
             *
             * Note that it only considers one, given, boundary.
             * So the particles outside the returned range crossed that boundary.
             * Particles in the returned range did not cross that boundary, but may still be outside for others.
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param exchangeType exchange describing the active boundary
             */
            template<typename T_Species>
            HINLINE auto getMapperFactory(T_Species& species, uint32_t exchangeType)
            {
                // First define a supercell interval for the whole local domain with guards
                auto mappingDescription = species.getCellDescription();
                auto beginSupercell = pmacc::DataSpace<simDim>::create(0);
                auto numSupercells = mappingDescription.getGridSuperCells();

                // Change to a single supercell along the active axis
                uint32_t const axis = pmacc::boundary::getAxis(exchangeType);
                numSupercells[axis] = 1;
                auto const offsetCells = getOffsetCells(species, exchangeType);
                auto const offsetSupercells
                    = (offsetCells + SuperCellSize::toRT()[axis] - 1) / SuperCellSize::toRT()[axis];
                auto const guardSupercells = mappingDescription.getGuardingSuperCells()[axis];
                if(pmacc::boundary::isMinSide(exchangeType))
                    beginSupercell[axis] = guardSupercells - 1 + offsetSupercells;
                else
                    beginSupercell[axis]
                        = mappingDescription.getGridSuperCells()[axis] - guardSupercells - offsetSupercells;
                return pmacc::IntervalMapperFactory<simDim>{beginSupercell, numSupercells};
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
