/* Copyright 2017-2021 Rene Widera
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

#include "picongpu/particles/functor/misc/DomainInfo.def"
#include "pmacc/mappings/simulation/Selection.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            namespace misc
            {
                struct DomainInfo
                {
                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE DomainInfo(uint32_t currentStep)
                    {
                        uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> const localCells = subGrid.getLocalDomain().size;

                        domInfo.local = subGrid.getLocalDomain();
                        domInfo.global = subGrid.getGlobalDomain();
                        domInfo.global.offset.y() += numSlides * localCells.y();
                    }

                    /** get cell offset of the supercell
                     *
                     * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                     * @tparam T_Acc alpaka accelerator type
                     *
                     * @param alpaka accelerator
                     * @param offset (in supercells, without any guards) to the
                     *         origin of the local domain
                     * @param configuration of the worker
                     */
                    template<typename T_WorkerCfg, typename T_Acc>
                    HDINLINE Domain operator()(
                        T_Acc const& acc,
                        DataSpace<simDim> const& localSupercellOffset,
                        T_WorkerCfg const& workerCfg) const
                    {
                        Domain currentDomInfo = domInfo;

                        DataSpace<simDim> const superCellToLocalOriginCellOffset(
                            localSupercellOffset * SuperCellSize::toRT());

                        auto cellInSuperCell
                            = DataSpaceOperations<simDim>::template map<SuperCellSize>(workerCfg.getWorkerIdx());

                        currentDomInfo.local.offset += superCellToLocalOriginCellOffset + cellInSuperCell;
                        currentDomInfo.global.offset += superCellToLocalOriginCellOffset + cellInSuperCell;
                        return currentDomInfo;
                    }

                private:
                    Domain domInfo;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
