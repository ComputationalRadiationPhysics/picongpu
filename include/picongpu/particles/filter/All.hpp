/* Copyright 2017-2023 Rene Widera
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

namespace picongpu
{
    namespace particles
    {
        namespace filter
        {
            namespace acc
            {
                //! check the particle handle
                struct All
                {
                    /** check particle handle
                     *
                     * @tparam T_Particle pmacc::Particles, type of the particle
                     * @tparam alpaka accelerator type
                     *
                     * @param worker lockstep worker
                     * @param particle  particle which is checked
                     * @return true if particle handle is valid, else false
                     */
                    template<typename T_Particle, typename T_Worker>
                    HDINLINE bool operator()(T_Worker const&, T_Particle const& particle)
                    {
                        return particle.isHandleValid();
                    }
                };

            } // namespace acc

            struct All
            {
                template<typename T_SpeciesType>
                struct apply
                {
                    using type = All;
                };

                /** create filter for the accelerator
                 *
                 * @tparam T_Worker lockstep::Worker, configuration of the worker
                 * @param offset (in superCells, without any guards) relative
                 *                        to the origin of the local domain
                 * @param configuration of the worker
                 */
                template<typename T_Worker>
                HDINLINE acc::All operator()(T_Worker const& worker, DataSpace<simDim> const&) const
                {
                    return acc::All{};
                }

                HINLINE static std::string getName()
                {
                    return std::string("all");
                }

                /** A filter is deterministic if the filter outcome is equal between evaluations. If so, set this
                 * variable to true, otherwise to false.
                 *
                 * Example: A filter were results depend on a random number generator must return false.
                 */
                static constexpr bool isDeterministic = true;
            };

        } // namespace filter
    } // namespace particles
} // namespace picongpu
