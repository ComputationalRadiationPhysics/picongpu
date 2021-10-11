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

#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"
#include "picongpu/particles/boundary/Parameters.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/functor/misc/Parametrized.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffsetRng.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <cmath>
#include <cstdint>
#include <limits>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Manipulator parameters, extend the basic version with temperature
            struct ThermalParameters : public Parameters
            {
                //! Boundary temperature in keV
                float_X temperature;
            };

            //! Functor to be applied to all particles in the active area
            struct ReflectThermalIfOutside : public functor::misc::Parametrized<ThermalParameters>
            {
                //! Name, required to be wrapped later
                static constexpr char const* name = "reflectThermalIfOutside";

                HDINLINE ReflectThermalIfOutside()
                    : energy((m_parameters.temperature * UNITCONV_keV_to_Joule) / UNIT_ENERGY)
                {
                }

                /** Process the current particle located in the given cell
                 *
                 * @param offsetToTotalOrigin offset of particle cell in the total domain
                 * @param rng random number generator with normal distribution
                 * @param particle handle of particle to process (can be used to change attribute values)
                 */
                template<typename T_Rng, typename T_Particle>
                HDINLINE void operator()(
                    DataSpace<simDim> const& offsetToTotalOrigin,
                    T_Rng& rng,
                    T_Particle& particle)
                {
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        if((offsetToTotalOrigin[d] < m_parameters.beginInternalCellsTotal[d])
                           || (offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d]))
                        {
                            // to check if its the right or left boundary which is crossed
                            bool rightBoundary = offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d];

                            float_X macroWeighting = particle[weighting_];
                            float_X macroEnergy = energy * macroWeighting;
                            float_X macroMass = attribute::getMass(macroWeighting, particle);
                            float_X standardDeviation
                                = static_cast<float_X>(math::sqrt(precisionCast<sqrt_X>(macroEnergy * macroMass)));

                            auto pos = particle[position_];
                            float3_X mom = particle[momentum_];

                            auto prevPos = pos; // values are not important, but just the same datatype as pos

                            Velocity velocity;
                            auto vel = velocity(mom, macroWeighting);

                            for(uint32_t i = 0; i < simDim; i++)
                            {
                                prevPos[i] = pos[d] - vel[i] * DELTA_T / cellSize[i];
                                mom[d] = rng() * standardDeviation; // do this here so it can be used in 2d and 3
                            }

                            // set sign to radiate the particle back into the simulation area on the reflected axiss
                            float_X sign = 1.0;
                            if(rightBoundary)
                                sign = -1.0;

                            mom[d] = sign * math::abs(mom[d]);

                            particle[momentum_] = mom;

                            // now the momentum was thermalized
                            auto velAfterThermal = velocity(mom, macroWeighting);

                            // get position ON the boundary to start the random radiation from there

                            // ddt: fraction of the time step DELTA_T the particle is outside of the boundary
                            float_X ddt;

                            if(rightBoundary)
                                // right boundary
                                ddt = pos[d] / (pos[d] + math::abs(prevPos[d]));
                            else
                                // left boundary
                                ddt = (1.0_X - pos[d]) / (prevPos[d] - pos[d]);

                            pos[d] = rightBoundary ? -std::numeric_limits<float_X>::epsilon()
                                                   : 1.0_X; // set pos on reflection axis ON the boundary

                            for(uint32_t i = 0; i < simDim; i++)
                            {
                                if(i != d)
                                    // pos on the boundary
                                    pos[i] -= vel[i] * DELTA_T * ddt / cellSize[i];

                                pos[i] += velAfterThermal[i] * DELTA_T * ddt / cellSize[i];
                            }

                            // TODO: check if the new position is inside the border on the other axis (!=d)  (i don't
                            // know if its relevant)

                            moveParticle(particle, pos);
                        }
                    }
                }

                //! Energy corresponding to temperature, in internal units
                float_X const energy;
            };

            //! Functor to apply thermal boundary to particle species
            template<>
            struct ApplyImpl<Kind::Thermal>
            {
                /** Apply thermal boundary conditions along the given outer boundary
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
                    // This is required for thermal boundaries until #3850 is resolved
                    if(!(getOffsetCells(species, exchangeType) > 0))
                        throw std::runtime_error("Thermal particle boundaries require a positive offset");

                    pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                    getInternalCellsTotal(species, exchangeType, &beginInternalCellsTotal, &endInternalCellsTotal);
                    ReflectThermalIfOutside::parameters().beginInternalCellsTotal = beginInternalCellsTotal;
                    ReflectThermalIfOutside::parameters().endInternalCellsTotal = endInternalCellsTotal;
                    ReflectThermalIfOutside::parameters().temperature = getTemperature(species, exchangeType);
                    auto const mapperFactory = getMapperFactory(species, exchangeType);
                    using Manipulator = manipulators::unary::
                        FreeTotalCellOffsetRng<ReflectThermalIfOutside, pmacc::random::distributions::Normal<float_X>>;
                    particles::manipulate<Manipulator, T_Species>(currentStep, mapperFactory);
                    auto const onlyProcessMustShiftSupercells = false;
                    species.shiftBetweenSupercells(mapperFactory, onlyProcessMustShiftSupercells);
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
