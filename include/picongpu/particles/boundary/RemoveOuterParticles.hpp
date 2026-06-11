/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/Manipulate.def"
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
            //! Parameters to be passed to particle boundary conditions
            struct AbsorbAllBoundariesParameters
            {
                /** Begin of the internal (relative to boundary) cells in total coordinates
                 *
                 * Particles to the left side are outside
                 */
                pmacc::DataSpace<simDim> beginInternalCellsTotal;

                /** End of the internal (relative to boundary) cells in total coordinates
                 *
                 * Particles equal or to the right side are outside
                 */
                pmacc::DataSpace<simDim> endInternalCellsTotal;
            };

            //! Functor to be applied to all particles in simulation
            struct AbsorbParticleIfOutsideAnyBoundary
                : public functor::misc::Parametrized<AbsorbAllBoundariesParameters>
            {
                //! Name, required to be wrapped later
                static constexpr char const* name = "absorbParticleIfOutsideAnyBoundary";

                /** Process the current particle located in the given cell, remove when outside any boundary
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
                            particle[multiMask_] = 0;
                }
            };

            /** Remove outer particles of the given species
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param currentStep current time iteration
             */
            template<typename T_Species>
            inline void removeOuterParticles(T_Species& species, uint32_t currentStep)
            {
                // Here we do basically same as in absorbing boundaries, but for all axes and the whole domain
                pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                getInternalCellsTotal(species, &beginInternalCellsTotal, &endInternalCellsTotal);
                AbsorbParticleIfOutsideAnyBoundary::parameters().beginInternalCellsTotal = beginInternalCellsTotal;
                AbsorbParticleIfOutsideAnyBoundary::parameters().endInternalCellsTotal = endInternalCellsTotal;
                using Manipulator = manipulators::unary::FreeTotalCellOffset<AbsorbParticleIfOutsideAnyBoundary>;
                particles::manipulate<Manipulator, T_Species>(currentStep);
                // Fill gaps to finalize deletion
                species.fillAllGaps();
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
