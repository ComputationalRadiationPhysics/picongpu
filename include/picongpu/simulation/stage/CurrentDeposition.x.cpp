/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include "picongpu/simulation/stage/CurrentDeposition.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldJ.kernel"
#include "picongpu/fields/currentDeposition/Deposit.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                template<typename T_SpeciesType, typename T_Area>
                struct CurrentDeposition
                {
                    using SpeciesType = T_SpeciesType;
                    using FrameType = typename SpeciesType::FrameType;

                    /** Compute current density created by a species in an area */
                    HINLINE void operator()(const uint32_t currentStep, FieldJ& fieldJ, pmacc::DataConnector& dc) const
                    {
                        auto species = dc.get<SpeciesType>(FrameType::getName());

                        /* Current deposition logic (for all schemes we implement) requires that a particle cannot pass
                         * more than a cell in a time step. For 2d this concerns only steps in x, y. This check is same
                         * as in particle pusher, but we do not require that pusher and current deposition are both
                         * enabled for a species, so check in both places.
                         */
                        constexpr auto dz
                            = (simDim == 3) ? sim.pic.getCellSize().z() : std::numeric_limits<float_X>::infinity();
                        constexpr auto minCellSize
                            = std::min({sim.pic.getCellSize().x(), sim.pic.getCellSize().y(), dz});
                        PMACC_CASSERT_MSG(
                            Particle_in_current_deposition_cannot_pass_more_than_1_cell_per_time_step____check_your_grid_param_file,
                            (sim.pic.getSpeedOfLight() * sim.pic.getDt() / minCellSize <= 1.0)
                                && sizeof(SpeciesType*) != 0);

                        using FrameType = typename SpeciesType::FrameType;
                        using ParticleCurrentSolver = typename pmacc::traits::Resolve<
                            typename pmacc::traits::GetFlagType<FrameType, current<>>::type>::type;

                        using FrameSolver = currentSolver::
                            ComputePerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize>;

                        using BlockArea = SuperCellDescription<
                            typename MappingDesc::SuperCellSize,
                            typename picongpu::traits::GetMargin<ParticleCurrentSolver>::LowerMargin,
                            typename picongpu::traits::GetMargin<ParticleCurrentSolver>::UpperMargin>;

                        using Strategy = currentSolver::traits::GetStrategy_t<FrameSolver>;

                        auto const depositionKernel = currentSolver::KernelComputeCurrent<BlockArea>{};

                        typename SpeciesType::ParticlesBoxType pBox = species->getDeviceParticlesBox();
                        FieldJ::DataBoxType jBox = fieldJ.getGridBuffer().getDeviceBuffer().getDataBox();
                        FrameSolver solver(sim.pic.getDt());

                        auto const deposit = currentSolver::Deposit<Strategy>{};
                        deposit.template execute<T_Area::value>(
                            species->getCellDescription(),
                            depositionKernel,
                            solver,
                            jBox,
                            pBox);
                    }
                };
            } // namespace detail

            void CurrentDeposition::operator()(uint32_t const step) const
            {
                using namespace pmacc;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());
                using SpeciesWithCurrentSolver =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
                meta::ForEach<
                    SpeciesWithCurrentSolver,
                    detail::CurrentDeposition<boost::mpl::_1, pmacc::mp_int<type::CORE + type::BORDER>>>
                    depositCurrent;
                depositCurrent(step, fieldJ, dc);
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
