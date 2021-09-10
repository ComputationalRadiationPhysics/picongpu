/* Copyright 2021 Sergei Bastrakov
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
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.hpp"

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            //! Functor to apply absorbing boundary to particle species
            template<>
            struct ApplyImpl<Kind::Absorbing>
            {
                //! Functor to be applied to all particles in the active area
                class AbsorbParticleIfOutside
                {
                public:
                    //! Some name is required
                    static constexpr char const* name = "absorbParticleIfOutside";

                    //! Construct an instance, the parameters must be passed via staticParameters()
                    AbsorbParticleIfOutside() : parameters(staticParameters())
                    {
                    }

                    /** Process the current particle located in the given cell
                     *
                     * @param offsetToTotalOrigin offset of particle cell in the total domain
                     * @param particle handle of particle to process (can be used to change attribute values)
                     */
                    template<typename T_Particle>
                    HDINLINE void operator()(DataSpace<simDim> const& offsetToTotalOrigin, T_Particle& particle)
                    {
                        for(uint32_t d = 0; d < simDim; d++)
                            if((offsetToTotalOrigin[d] < parameters.beginInternalCellsTotal[d])
                               || (offsetToTotalOrigin[d] >= parameters.endInternalCellsTotal[d]))
                                particle[multiMask_] = 0;
                    }

                    //! Parameters to be passed from the host side
                    struct Parameters
                    {
                        /** Begin of the internal (not-absorbed) cells in total coordinates
                         *
                         * Particles to the left side will be absorbed
                         */
                        pmacc::DataSpace<simDim> beginInternalCellsTotal;

                        /** End of the internal (not-absorbed) cells in total coordinates
                         *
                         * Particles equal or to the right side will be absorbed
                         */
                        pmacc::DataSpace<simDim> endInternalCellsTotal;
                    };

                    /** Pass the parameters from the host side by changing this value_comp
                     *
                     * It is a poor man's way to do it, but otherwise that would be awkward to implement
                     */
                    static Parameters& staticParameters()
                    {
                        static auto parametersValue = Parameters{};
                        return parametersValue;
                    }

                private:
                    Parameters parameters;
                };

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
                    // For no offset, nothing has to be done as the particles in GUARD will be just removed later
                    auto offsetCells = getOffsetCells(species, exchangeType);
                    if(offsetCells == 0)
                        return;

                    /* The rest of this function is not optimal performance-wise.
                     * However it is only used when a user set a positive offset, so tolerable.
                     * It processes all particles in manipulate and fillAllGaps() instead of working on the active area
                     * specifically. Currently it would also go over several times if multiple boundaries are
                     * absorbing.
                     */
                    pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                    getInternalCellsTotal(species, exchangeType, &beginInternalCellsTotal, &endInternalCellsTotal);
                    AbsorbParticleIfOutside::staticParameters().beginInternalCellsTotal = beginInternalCellsTotal;
                    AbsorbParticleIfOutside::staticParameters().endInternalCellsTotal = endInternalCellsTotal;
                    using Manipulator = manipulators::unary::FreeTotalCellOffset<AbsorbParticleIfOutside>;
                    particles::manipulate<Manipulator, T_Species>(currentStep);
                    // Fill gaps to finalize deletion
                    species.fillAllGaps();
                }
            };

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
