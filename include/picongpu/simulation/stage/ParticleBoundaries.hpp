/* Copyright 2021-2023 Sergei Bastrakov
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

#include <boost/program_options.hpp>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for setting up particle boundaries for species with a pusher
             *
             * Allows overwriting default boundaries via command-line for those species.
             * This stage does not apply boudaries by itself, but is needed to propagate command-line parameters
             */
            class ParticleBoundaries
            {
            public:
                /** Register program options for particle boundaries
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(boost::program_options::options_description& desc);

                /** Initialize particle boundaries stage
                 *
                 * Sets boundary kind values for all affected species.
                 */
                void init();
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
