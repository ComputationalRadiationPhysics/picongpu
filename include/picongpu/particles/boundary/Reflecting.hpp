/* Copyright 2021 Sergei Bastrakov, Lennert Sprenger
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

#include "picongpu/particles/MoveParticle.hpp"
#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"
#include "picongpu/particles/boundary/Parameters.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/functor/misc/Parametrized.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.hpp"

#include <cstdint>
#include <limits>

namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            struct ReflectParticleIfOutside : public functor::misc::Parametrized<Parameters>
            {
                //! Name, required to be wrapped later
                static constexpr char const* name = "reflectParticleIfOutside";

                /** Process the current particle located in the given cell
                 *
                 * @param offsetToTotalOrigin offset of particle cell in the total domain
                 * @param particle handle of particle to process (can be used to change attribute values)
                 */
                template<typename T_Particle>
                HDINLINE void operator()(DataSpace<simDim> const& offsetToTotalOrigin, T_Particle& particle)
                {
                    for(uint32_t d = 0; d < simDim; d++)
                        if((offsetToTotalOrigin[d] < m_parameters.beginInternalCellsTotal[d])
                           || (offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d]))
                        {
                            auto pos = particle[position_]; // floatD_X pos

                            if(offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d])
                            {
                                /*
                                 * substract epsilon so we are definitly in the cell left of the current
                                 * cell, because that could happen if pos = 0 or pos < epsilon.
                                 * To set the position to 0.9 of the left cell relative to the current cell,
                                 * pos needs to be set to -0.1.
                                 */
                                pos[d] = -pos[d] - std::numeric_limits<float_X>::epsilon();
                            }
                            else
                                /*
                                 * no correction is needed here, because if the particle just barly crossed the left
                                 * border pos[d] is about 0.9 in the current cell. In order to be very small ( <
                                 * epsilon ) the particle needs to be on the left side of the cell and that would mean
                                 * that the particle traveled a whole cell which should not be possible with the with
                                 * the CFL-condition.
                                 */
                                pos[d] = 2.0_X - pos[d];

                            particle[momentum_][d] = -particle[momentum_][d];
                            moveParticle(particle, pos);
                        }
                }
            };

            //! Functor to apply reflecting boundary condition to particle species
            template<>
            struct ApplyImpl<Kind::Reflecting>
            {
                /** Apply reflecting boundary conditions along the given outer boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                void operator()(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
                {
                    pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                    getInternalCellsTotal(species, exchangeType, &beginInternalCellsTotal, &endInternalCellsTotal);
                    ReflectParticleIfOutside::parameters().beginInternalCellsTotal = beginInternalCellsTotal;
                    ReflectParticleIfOutside::parameters().endInternalCellsTotal = endInternalCellsTotal;
                    auto const mapperFactory = getMapperFactory(species, exchangeType);
                    using Manipulator = manipulators::unary::FreeTotalCellOffset<ReflectParticleIfOutside>;
                    particles::manipulate<Manipulator, T_Species>(currentStep, mapperFactory);

                    /* After reflection is applied, some particles can require movement between supercells.
                     * We have not set mustShift for those supercells, so shift has to process all supercells
                     */
                    auto const onlyProcessMustShiftSupercells = false;
                    species.shiftBetweenSupercells(mapperFactory, onlyProcessMustShiftSupercells);
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
