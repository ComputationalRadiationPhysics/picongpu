/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
