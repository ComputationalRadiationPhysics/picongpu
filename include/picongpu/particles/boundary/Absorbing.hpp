/* Copyright 2021-2022 Sergei Bastrakov
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
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.hpp"

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            namespace detail
            {
                //! Functor to be applied to all particles in the active area
                struct AbsorbParticleIfOutside : public functor::misc::Parametrized<Parameters>
                {
                    //! Name, required to be wrapped later
                    static constexpr char const* name = "absorbParticleIfOutside";

                    /** Process the current particle located in the given cell
                     *
                     * @param offsetToTotalOrigin offset of particle cell in the total domain
                     * @param particle handle of particle to process (can be used to change attribute values)
                     */
                    template<typename T_Particle>
                    HDINLINE void operator()(DataSpace<simDim> const& offsetToTotalOrigin, T_Particle& particle)
                    {
                        if((offsetToTotalOrigin[m_parameters.axis] < m_parameters.beginInternalCellsTotal)
                           || (offsetToTotalOrigin[m_parameters.axis] >= m_parameters.endInternalCellsTotal))
                            particle[multiMask_] = 0;
                    }
                };

                /** Remove particles of the given species that are outer wrt the given boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                HINLINE void removeOuterParticles(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
                {
                    pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                    getInternalCellsTotal(species, exchangeType, &beginInternalCellsTotal, &endInternalCellsTotal);
                    auto const axis = pmacc::boundary::getAxis(exchangeType);
                    AbsorbParticleIfOutside::parameters().axis = axis;
                    AbsorbParticleIfOutside::parameters().beginInternalCellsTotal = beginInternalCellsTotal[axis];
                    AbsorbParticleIfOutside::parameters().endInternalCellsTotal = endInternalCellsTotal[axis];
                    auto const mapperFactory = getMapperFactory(species, exchangeType);
                    using Manipulator = manipulators::unary::FreeTotalCellOffset<AbsorbParticleIfOutside>;
                    particles::manipulate<Manipulator, T_Species>(currentStep, mapperFactory);
                    species.fillGaps(mapperFactory);
                }

            } // namespace detail

            //! Functor to apply absorbing boundary to particle species
            template<>
            struct ApplyImpl<Kind::Absorbing>
            {
                /** Apply absorbing boundary conditions along the given outer boundary
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
                    detail::removeOuterParticles(species, exchangeType, currentStep);
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
