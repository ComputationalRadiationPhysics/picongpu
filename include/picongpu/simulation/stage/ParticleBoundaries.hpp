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

#include "picongpu/particles/Particles.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>

#include <boost/program_options.hpp>

#include <cstdint>
#include <string>
#include <vector>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                /** Functor to set boundary kind for the given species via command-line parameters
                 *
                 * @tparam T_Species particle species type
                 */
                template<typename T_Species>
                class ParticleBoundariesCommandLine
                {
                public:
                    //! Create particle boundaries functor
                    ParticleBoundariesCommandLine() : prefix(T_Species::FrameType::getName())
                    {
                    }

                    /** Register command-line options
                     *
                     * Done as operator() to simplify invoking with pmacc::meta::ForEach
                     */
                    void operator()(po::options_description& desc)
                    {
                        auto example = std::string{"example: --" + prefix + "_boundary absorbing periodic"};
                        if(simDim == 3)
                            example += " periodic";
                        desc.add_options()(
                            (prefix + "_boundary").c_str(),
                            po::value<std::vector<std::string>>(&(kindNames()))->multitoken(),
                            std::string(
                                "Boundary kinds for species '" + prefix
                                + "' for each axis. "
                                  "Supported values: default (matching --periodic values), periodic, absorbing"
                                  "\n"
                                + example)
                                .c_str());
                    }

                    /** Set boundary kind of T_Species according to command-line options
                     *
                     * Done as operator() to simplify invoking with pmacc::meta::ForEach
                     */
                    void operator()()
                    {
                        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
                        {
                            // for each planar direction: left right top bottom back front
                            if(FRONT % i == 0)
                            {
                                const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);
                                uint32_t axis = 0; // x(0) y(1) z(2)
                                for(uint32_t d = 0; d < simDim; d++)
                                    if(relDir[d] != 0)
                                        axis = d;
                                /* For now we do not support any type not matching --periodic values.
                                 * So just check that the user-provided type matches the default one.
                                 */
                                auto const kindName = kindNames()[axis];
                                bool isInvalidPeriodic
                                    = ((kindName == "periodic")
                                       && (T_Species::boundaryKind()[axis] != particles::boundary::Kind::Periodic));
                                bool isInvalidAbsorbing
                                    = ((kindName == "absorbing")
                                       && (T_Species::boundaryKind()[axis] != particles::boundary::Kind::Absorbing));
                                if(isInvalidPeriodic || isInvalidAbsorbing)
                                {
                                    throw std::runtime_error(
                                        "Boundary kind for species '" + prefix + "' and axis " + std::to_string(axis)
                                        + " is not compatible with --periodic value");
                                }
                            }
                        }
                    }

                private:
                    //! Names of boundary kinds for all axes
                    static std::vector<std::string>& kindNames()
                    {
                        static auto names = std::vector<std::string>(simDim, "default");
                        return names;
                    }

                    //! Prefix for the given species
                    std::string prefix;
                };

            } // namespace detail

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
                void registerHelp(po::options_description& desc)
                {
                    processSpecies(desc);
                }

                /** Initialize particle boundaries stage
                 *
                 * Sets boundary kind values for all affected species.
                 */
                void init()
                {
                    processSpecies();
                }

            private:
                //! Only allow customization for species with pusher
                using SpeciesWithPusher =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
                //! Functor to process all affected species
                meta::ForEach<SpeciesWithPusher, detail::ParticleBoundariesCommandLine<bmpl::_1>> processSpecies;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
