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


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            struct ReflectParticleIfOutside : public functor::misc::Parametrized<Parameters>
            {
                //! some name is required
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
                            auto pos = particle[position_];

                            if(offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d])
                                pos[d] = -pos[d];
                            else
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
