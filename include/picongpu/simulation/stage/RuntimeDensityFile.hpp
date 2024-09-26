/* Copyright 2022-2023 Sergei Bastrakov
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/Particles.hpp"
#include "picongpu/particles/boundary/Kind.hpp"
#include "picongpu/particles/densityProfiles/FromOpenPMDImpl.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

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
                /** Functor to set runtime density file for the given species via command-line parameters
                 *
                 * @tparam T_Species particle species type
                 */
                template<typename T_Species>
                class RuntimeDensityFileCommandLine
                {
                public:
                    //! Create runtime density file functor
                    RuntimeDensityFileCommandLine() : prefix(T_Species::FrameType::getName())
                    {
                    }

                    /** Register command-line options
                     *
                     * Done as operator() to simplify invoking with pmacc::meta::ForEach
                     */
                    void operator()(po::options_description& desc)
                    {
                        // Density from openPMD is conditionally enabled, so use same condition
#if(ENABLE_OPENPMD == 1)
                        auto* filename = &densityProfiles::RuntimeDensityFile<T_Species>::get();
                        desc.add_options()(
                            (prefix + "_runtimeDensityFile").c_str(),
                            po::value<std::string>(filename),
                            std::string(
                                "Runtime density file name for species '" + prefix
                                + "'\n. Only has effect when FromOpenPMDImpl<Param> density is used for the species "
                                  "and its Param::filename is empty.\n")
                                .c_str());
#endif
                    }

                private:
                    //! Prefix for the given species
                    std::string prefix;
                };

            } // namespace detail

            /** Functor for setting up runtime path for density from file
             *
             * Affects invocations of FromOpenPMDImpl<Param> with empty Param::filename.
             * This stage allows to effectively set its value in runtime using a prefix of species it is applied to.
             * In case there are several invocations for the same species, the same runtime parameter is used for all.
             *
             * Note that this is merely a runtime convenience hook for what normally is a compile-time variable.
             * It does not perform calculations by itself, nor affects other parameters or logic of FromOpenPMDImpl.
             */
            class RuntimeDensityFile
            {
            public:
                /** Register program options for runtime density file
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
                    processSpecies(desc);
                }

                /** Initialize runtime density file stage
                 *
                 * Sets default paths values for all affected species.
                 */
                void init()
                {
                    // All work is already done, this method exists for consistency with other similar stages
                }

            private:
                //! Functor to process all species
                meta::ForEach<VectorAllSpecies, detail::RuntimeDensityFileCommandLine<boost::mpl::_1>> processSpecies;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
