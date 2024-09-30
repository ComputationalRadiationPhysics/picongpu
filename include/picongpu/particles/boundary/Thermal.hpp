/* Copyright 2021-2023 Sergei Bastrakov, Lennert Sprenger
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

#pragma once

#include "picongpu/algorithms/Velocity.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/Manipulate.def"
#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"
#include "picongpu/particles/boundary/Parameters.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/functor/misc/Parametrized.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffsetRng.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/boundary/Utility.hpp>
#include <pmacc/random/distributions/distributions.hpp>

#include <cmath>
#include <cstdint>
#include <limits>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Parameters for thermal boundary implementation
            struct ThermalParameters
            {
                //! Boundary temperature in keV
                float_X temperature;

                //! Axis of the active thermal boundary
                uint32_t axis;

                /** Begin of the internal (relative to all boundaries) cells in total coordinates
                 *
                 * Particles to the left side are outside
                 */
                pmacc::DataSpace<simDim> beginInternalCellsTotalAllBoundaries;

                /** End of the internal (relative to all boundaries) cells in total coordinates
                 *
                 * Particles equal or to the right side are outside
                 */
                pmacc::DataSpace<simDim> endInternalCellsTotalAllBoundaries;
            };

            //! Functor to be applied to all particles in the active area
            struct ReflectThermalIfOutside : public functor::misc::Parametrized<ThermalParameters>
            {
                //! Name, required to be wrapped later
                static constexpr char const* name = "reflectThermalIfOutside";

                HINLINE ReflectThermalIfOutside()
                    : energy((m_parameters.temperature * sim.si.conv().eV2Joule(1.0e3)) / sim.unit.energy())
                {
                }

                /** Process the current particle located in the given cell
                 *
                 * Unlike other boundary implementations, thermal has to account for all boundary crossings.
                 * The reason is, it samples a new momentum and thus effectively erases the particle move history.
                 * After a particle is processed, it will be located inside, and with an inside-pointing momentum,
                 * with respect to all boundaries.
                 * Doing otherwise would be dangerous, as then it could happen that at the same time a particle
                 * is crossing by its cell index, but both current and "old" positions (calculated with rewritten
                 * momentum) are on the same side of the boundary.
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
                    // > 0 means we crossed right boundary, < 0 means crossed left boundary, 0 means didn't cross
                    auto crossedBoundary = pmacc::DataSpace<simDim>::create(0);
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        if(offsetToTotalOrigin[d] < m_parameters.beginInternalCellsTotalAllBoundaries[d])
                            crossedBoundary[d] = -1;
                        else if(offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotalAllBoundaries[d])
                            crossedBoundary[d] = 1;
                        else
                            crossedBoundary[d] = 0;
                    }
                    // Only apply thermal BC logic when an active thermal boundary was crossed
                    if(crossedBoundary[m_parameters.axis])
                    {
                        // Save velocity before thermalization
                        float_X const macroWeighting = particle[weighting_];
                        float_X const mass = picongpu::traits::attribute::getMass(macroWeighting, particle);
                        float3_X mom = particle[momentum_];
                        auto velocity = Velocity{};
                        auto velBeforeThermal = velocity(mom, mass);

                        /* Sample momentum and for crossing particles make it point inwards wrt all boundaries.
                         * Here we have to check all boundaries and not just the first-crossed one.
                         * This is to avoid particles getting outside the boundary e.g. in case of applying another BC.
                         * For 2d do not thermalize in z.
                         */
                        float_X const macroEnergy = energy * macroWeighting;
                        float_X const macroMass = picongpu::traits::attribute::getMass(macroWeighting, particle);
                        float_X const standardDeviation
                            = static_cast<float_X>(math::sqrt(precisionCast<sqrt_X>(macroEnergy * macroMass)));
                        for(uint32_t d = 0; d < simDim; d++)
                        {
                            mom[d] = rng() * standardDeviation;
                            if(crossedBoundary[d] > 0)
                                mom[d] = -math::abs(mom[d]);
                            else if(crossedBoundary[d] < 0)
                                mom[d] = math::abs(mom[d]);
                        }
                        particle[momentum_] = mom;

                        // Find the first time the particle crossed any boundary, in internal units
                        float_X timeAfterFirstBoundaryCross = 0._X;
                        auto pos = particle[position_];
                        auto const epsilon = std::numeric_limits<float_X>::epsilon();
                        for(uint32_t d = 0; d < simDim; d++)
                        {
                            /* Add epsilon to avoid potential divisions by 0.
                             * We clump the position later so it is no problem that timeAfterCross is slightly below
                             * its analytical value.
                             */
                            auto timeAfterCross = 0.0_X;
                            if(crossedBoundary[d] > 0)
                                timeAfterCross = pos[d] * sim.pic.getCellSize()[d] / (velBeforeThermal[d] + epsilon);
                            else if(crossedBoundary[d] < 0)
                                timeAfterCross
                                    = (1.0_X - pos[d]) * sim.pic.getCellSize()[d] / (-velBeforeThermal[d] - epsilon);
                            timeAfterFirstBoundaryCross = math::max(timeAfterFirstBoundaryCross, timeAfterCross);
                        }

                        /* Replace movement in the last timeAfterFirstBoundaryCross with new velocity.
                         * Afterwards take care the particle is safely in the internal side for all boundaries.
                         */
                        auto velAfterThermal = velocity(mom, mass);
                        for(uint32_t d = 0; d < simDim; d++)
                        {
                            pos[d] += (velAfterThermal[d] - velBeforeThermal[d]) * timeAfterFirstBoundaryCross
                                / sim.pic.getCellSize()[d];
                            if((crossedBoundary[d] > 0) && (pos[d] >= -epsilon))
                                pos[d] = -epsilon;
                            else if((crossedBoundary[d] < 0) && (pos[d] <= 1.0_X))
                                pos[d] = 1.0_X;
                            /* When a particle is crossing multiple boundaries at once, it could (rarely) happen that
                             * it is jumping a cell as the result of this procedure.
                             * For example, consider a particle that is in a near-boundary cell in both x and y.
                             * This particle could be at the center of cell in x and near the border in y.
                             * Now, consider what happens when it moves diagonally in x, y and crosses both boundaries.
                             * Since y was very close to threshold, timeAfterFirstBoundaryCross would be almost
                             * sim.pic.getDt(). So we will revert almost the whole movement and effectively do another
                             * push. Then it can happen that in x we move sufficiently to end up in 2 cells from the
                             * current one. There is nothing we can easily fix here, so just clump the position to be
                             * in valid range. This case is rare and so should not disrupt the physics. This clump also
                             * guards against the case when the original momentum of the particle was somehow modified
                             * non-consistently with a position. For example, when the particle crossed by its cell
                             * index, but (current position - v * dt) is on the same side as current position.
                             */
                            pos[d] = math::max(-1.0_X, math::min(pos[d], 2.0_X * (1.0_X - epsilon)));
                        }
                        moveParticle(particle, pos);
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
                    // Positive offset is required for thermal boundaries until #3850 is resolved
                    if(getOffsetCells(species, exchangeType) <= 0)
                        throw std::runtime_error("Thermal particle boundaries require a positive offset");

                    // Here we need area wrt all boundaries, not just the current one
                    pmacc::DataSpace<simDim> beginInternalCellsTotalAllBoundaries, endInternalCellsTotalAllBoundaries;
                    getInternalCellsTotal(
                        species,
                        &beginInternalCellsTotalAllBoundaries,
                        &endInternalCellsTotalAllBoundaries);
                    ReflectThermalIfOutside::parameters().beginInternalCellsTotalAllBoundaries
                        = beginInternalCellsTotalAllBoundaries;
                    ReflectThermalIfOutside::parameters().endInternalCellsTotalAllBoundaries
                        = endInternalCellsTotalAllBoundaries;
                    ReflectThermalIfOutside::parameters().temperature = getTemperature(species, exchangeType);
                    ReflectThermalIfOutside::parameters().axis = pmacc::boundary::getAxis(exchangeType);
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
