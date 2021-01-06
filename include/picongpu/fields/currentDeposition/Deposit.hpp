/* Copyright 2020-2021 Rene Widera
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

#include "picongpu/particles/traits/GetCurrentSolver.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/StrideMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>


namespace picongpu
{
    namespace currentSolver
    {
        /** Executes the current deposition kernel
         *
         * @tparam T_Strategy Used strategy to reduce the scattered data [currentSolver::strategy]
         * @tparam T_Sfinae Optional specialization
         */
        template<typename T_Strategy, typename T_Sfinae = void>
        struct Deposit;

        template<typename T_Strategy>
        struct Deposit<T_Strategy, typename std::enable_if<T_Strategy::stridedMapping>::type>
        {
            /** Execute the current deposition with a checker board
             *
             * The stride between the supercells for the checker board will be automatically
             * adjusted, based on the species shape.
             */
            template<
                uint32_t T_area,
                uint32_t T_numWorkers,
                typename T_CellDescription,
                typename T_DepositionKernel,
                typename T_FrameSolver,
                typename T_JBox,
                typename T_ParticleBox>
            void execute(
                T_CellDescription const& cellDescription,
                T_DepositionKernel const& depositionKernel,
                T_FrameSolver const& frameSolver,
                T_JBox const& jBox,
                T_ParticleBox const& parBox) const
            {
                /* The needed stride for the stride mapper depends on the stencil width.
                 * If the upper and lower margin of the stencil fits into one supercell
                 * a double checker board (stride 2) is needed.
                 * The round up sum of margins is the number of supercells to skip.
                 */
                using MarginPerDim = typename pmacc::math::CT::add<
                    typename GetMargin<typename T_FrameSolver::ParticleAlgo>::LowerMargin,
                    typename GetMargin<typename T_FrameSolver::ParticleAlgo>::UpperMargin>::type;
                using MaxMargin = typename pmacc::math::CT::max<MarginPerDim>::type;
                using SuperCellMinSize = typename pmacc::math::CT::min<SuperCellSize>::type;

                /* number of supercells which must be skipped to avoid overlapping areas
                 * between different blocks in the kernel
                 */
                constexpr uint32_t skipSuperCells
                    = (MaxMargin::value + SuperCellMinSize::value - 1u) / SuperCellMinSize::value;
                StrideMapping<
                    T_area,
                    skipSuperCells + 1u, // stride 1u means each supercell is used
                    MappingDesc>
                    mapper(cellDescription);

                do
                {
                    PMACC_KERNEL(depositionKernel)
                    (mapper.getGridDim(), T_numWorkers)(jBox, parBox, frameSolver, mapper);
                } while(mapper.next());
            }
        };

        template<typename T_Strategy>
        struct Deposit<T_Strategy, typename std::enable_if<!T_Strategy::stridedMapping>::type>
        {
            /** Execute the current deposition for each supercell
             *
             * All supercells will be processed in parallel.
             */
            template<
                uint32_t T_area,
                uint32_t T_numWorkers,
                typename T_CellDescription,
                typename T_DepositionKernel,
                typename T_FrameSolver,
                typename T_JBox,
                typename T_ParticleBox>
            void execute(
                T_CellDescription const& cellDescription,
                T_DepositionKernel const& depositionKernel,
                T_FrameSolver const& frameSolver,
                T_JBox const& jBox,
                T_ParticleBox const& parBox) const
            {
                AreaMapping<T_area, MappingDesc> mapper(cellDescription);

                PMACC_KERNEL(depositionKernel)(mapper.getGridDim(), T_numWorkers)(jBox, parBox, frameSolver, mapper);
            }
        };

    } // namespace currentSolver
} // namespace picongpu
