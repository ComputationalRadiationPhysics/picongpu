/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch
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
#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/LaserPhysics.def"
#include "picongpu/fields/laserProfiles/profiles.hpp"

#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/mappings/threads/WorkerCfg.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>

#include <cmath>


namespace picongpu
{
    namespace fields
    {
        /** compute the electric field of the laser
         *
         * @tparam T_numWorkers number of workers
         * @tparam T_LaserPlaneSizeInSuperCell number of cells per dimension which
         *  initialize the laser (size must be less or equal than the supercell size)
         */
        template<uint32_t T_numWorkers, typename T_LaserPlaneSizeInSuperCell>
        struct KernelLaser
        {
            template<typename T_Acc, typename T_LaserFunctor>
            DINLINE void operator()(T_Acc const& acc, T_LaserFunctor laserFunctor) const
            {
                using LaserPlaneSizeInSuperCell = T_LaserPlaneSizeInSuperCell;
                using LaserFunctor = T_LaserFunctor;

                PMACC_CASSERT_MSG(
                    __LaserPlaneSizeInSuperCell_y_must_be_less_or_equal_than_SuperCellSize_y,
                    LaserPlaneSizeInSuperCell::y::value <= SuperCellSize::y::value);

                constexpr uint32_t planeSize = pmacc::math::CT::volume<LaserPlaneSizeInSuperCell>::type::value;
                PMACC_CONSTEXPR_CAPTURE uint32_t numWorkers = T_numWorkers;

                const uint32_t workerIdx = cupla::threadIdx(acc).x;

                // offset of the superCell (in cells, without any guards) to the origin of the local domain

                DataSpace<simDim> localSuperCellOffset = DataSpace<simDim>(cupla::blockIdx(acc));

                // add not handled supercells from LaserFunctor::Unitless::initPlaneY
                localSuperCellOffset.y() += LaserFunctor::Unitless::initPlaneY / SuperCellSize::y::value;

                uint32_t cellOffsetInSuperCellFromInitPlaneY
                    = LaserFunctor::Unitless::initPlaneY % SuperCellSize::y::value;

                mappings::threads::ForEachIdx<mappings::threads::IdxConfig<planeSize, numWorkers>>{
                    workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                    auto accLaserFunctor
                        = laserFunctor(acc, localSuperCellOffset, mappings::threads::WorkerCfg<numWorkers>{workerIdx});

                    /* cell index within the superCell */
                    DataSpace<simDim> cellIdxInSuperCell
                        = DataSpaceOperations<simDim>::template map<LaserPlaneSizeInSuperCell>(linearIdx);
                    cellIdxInSuperCell.y() += cellOffsetInSuperCellFromInitPlaneY;

                    accLaserFunctor(acc, cellIdxInSuperCell);
                });
            }
        };

        /** Laser init in a single xz plane */
        struct LaserPhysics
        {
            void operator()(uint32_t currentStep) const
            {
                /* The laser can be initialized in the plane of the first cell or
                 * any later x-z plane inside the simulation. Initializing the
                 * laser in planes inside the simulation corresponds to an
                 * evaluation of the field at negatively shifted time.
                 */
                constexpr float_X laserTimeShift
                    = laserProfiles::Selected::Unitless::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

                const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);

                /* Disable laser if
                 * - init time of laser is over or
                 * - we have periodic boundaries in Y direction or
                 * - we already performed a slide
                 */
                bool const laserNone = (laserProfiles::Selected::Unitless::INIT_TIME == float_X(0.0));
                bool const laserInitTimeOver
                    = ((currentStep * DELTA_T - laserTimeShift) >= laserProfiles::Selected::Unitless::INIT_TIME);
                bool const topBoundariesArePeriodic
                    = (Environment<simDim>::get().GridController().getCommunicationMask().isSet(TOP));
                bool const boxHasSlided = (numSlides != 0);

                bool const disableLaser = laserNone || laserInitTimeOver || topBoundariesArePeriodic || boxHasSlided;
                if(!disableLaser)
                {
                    PMACC_VERIFY_MSG(
                        laserProfiles::Selected::Unitless::initPlaneY
                            < static_cast<uint32_t>(Environment<simDim>::get().SubGrid().getLocalDomain().size.y()),
                        "initPlaneY must be located in the top GPU");

                    // laser is disabled e.g. laserNone
                    constexpr bool isLaserDisabled = laserProfiles::Selected::Unitless::INIT_TIME == 0.0_X;
                    constexpr bool isLaserInitInFirstCell = laserProfiles::Selected::Unitless::initPlaneY == 0;
                    // X + 1 is a workaround to avoid warning: pointless comparison of unsigned integer with zero
                    constexpr bool isInitPlaneYOutsideOfAbsorber
                        = laserProfiles::Selected::Unitless::initPlaneY + 1 > absorber::numCells[1][0] + 1;
                    PMACC_CASSERT_MSG(
                        __initPlaneY_needs_to_be_greater_than_the_top_absorber_cells_or_zero,
                        isLaserDisabled || isLaserInitInFirstCell || isInitPlaneYOutsideOfAbsorber);

                    /* Calculate how many neighbors to the left we have
                     * to initialize the laser in the E-Field
                     *
                     * Example: Yee needs one neighbor to perform dB = curlE
                     *            -> initialize in y=0 plane
                     *          A second order solver could need 2 neighbors left:
                     *            -> initialize in y=0 and y=1 plane
                     *
                     * Question: Why do other codes initialize the B-Field instead?
                     * Answer:   Because our fields are defined on the lower cell side
                     *           (C-Style ftw). Therefore, our curls (for example Yee)
                     *           are shifted nabla+ <-> nabla- compared to Fortran codes
                     *           (in other words: curlLeft <-> curlRight)
                     *           for E and B.
                     *           For this reason, we have to initialize E instead of B.
                     *
                     * Problem: that's still not our case. For example our Yee does a
                     *          dE = curlLeft(B) - therefor, we should init B, too.
                     *
                     *
                     *  @todo: might also lack temporal offset since our formulas are E(x,z,t) instead of E(x,y,z,t)
                     *  `const int max_y_neighbors = Get<fields::Solver::OffsetOrigin_E, 1 >::value;`
                     *
                     * @todo Right now, the phase could be wrong ( == is cloned)
                     *       @see LaserPhysics.hpp
                     *
                     * @todo What about the B-Field in the second plane?
                     *
                     */
                    constexpr int laserInitCellsInY = 1;

                    using LaserPlaneSizeInSuperCells = typename pmacc::math::CT::AssignIfInRange<
                        typename SuperCellSize::vector_type,
                        bmpl::integral_c<uint32_t, 1>, /* y direction */
                        bmpl::integral_c<int, laserInitCellsInY>>::type;

                    DataSpace<simDim> gridBlocks
                        = Environment<simDim>::get().SubGrid().getLocalDomain().size / SuperCellSize::toRT();
                    // use the one supercell in y to initialize the laser plane
                    gridBlocks.y() = 1;

                    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                        pmacc::math::CT::volume<LaserPlaneSizeInSuperCells>::type::value>::value;

                    PMACC_KERNEL(KernelLaser<numWorkers, LaserPlaneSizeInSuperCells>{})
                    (gridBlocks, numWorkers)(laserProfiles::Selected(currentStep));
                }
            }
        };
    } // namespace fields
} // namespace picongpu
