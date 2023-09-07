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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/Particles.hpp"
#include "picongpu/particles/boundary/Kind.hpp"

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
                /** Functor to set boundary options for the given species via command-line parameters
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
                        if constexpr(simDim == 3)
                            example += " reflecting";
                        desc.add_options()(
                            (prefix + "_boundary").c_str(),
                            po::value<std::vector<std::string>>(&(kindNames()))->multitoken(),
                            std::string(
                                "Boundary kinds for species '" + prefix
                                + "' for each axis. "
                                  "Supported values: default (matching --periodic values), periodic, absorbing, "
                                  "reflecting, thermal\n"
                                + example)
                                .c_str());
                        desc.add_options()(
                            (prefix + "_boundaryOffset").c_str(),
                            po::value<std::vector<int32_t>>(&(offsets()))->multitoken(),
                            std::string(
                                "Boundary offsets inwards from global domain boundary for species '" + prefix
                                + "' for each axis. "
                                  "Periodic boundaries only allow 0 offsets, other kinds support non-negative offsets")
                                .c_str());
                        desc.add_options()(
                            (prefix + "_boundaryTemperature").c_str(),
                            po::value<std::vector<float_X>>(&(temperatures()))->multitoken(),
                            std::string(
                                "Boundary temperatures (only affects thermal boundaries) for species '" + prefix
                                + "' for each axis, in keV.")
                                .c_str());
                    }

                    /** Set boundary description of T_Species according to command-line options
                     *
                     * Done as operator() to simplify invoking with pmacc::meta::ForEach
                     */
                    void operator()()
                    {
                        static bool validateUserInputs = true;
                        if(validateUserInputs)
                        {
                            validateUserInputs = false;
                            PMACC_VERIFY_MSG(
                                kindNames().size() <= 3,
                                std::string("Invalid number of particle boundary kinds for species '") + prefix
                                    + "'.");
                            PMACC_VERIFY_MSG(
                                offsets().size() <= 3,
                                std::string("Invalid number of particle boundary offsets for species '") + prefix
                                    + "'.");
                            PMACC_VERIFY_MSG(
                                temperatures().size() <= 3,
                                std::string("Invalid number of particle boundary temperatures for species '") + prefix
                                    + "'.");

                            for(uint32_t d = 0; d < simDim; d++)
                            {
                                auto const errorString
                                    = std::string{"for species '" + prefix + "' and axis " + std::to_string(d)};
                                int32_t offset = offsets()[d];
                                if(offset < 0)
                                    throw std::runtime_error(
                                        "Negative boundary offset " + errorString + " is not supported");
                                T_Species::boundaryDescription()[d].offset = offset;
                                float_X temperature = temperatures()[d];
                                if(temperature < 0.0_X)
                                    throw std::runtime_error(
                                        "Negative boundary temperature " + errorString + " is not supported");
                                T_Species::boundaryDescription()[d].temperature = temperature;

                                auto const kindName = kindNames()[d];
                                if(kindName == "reflecting")
                                {
                                    if(T_Species::boundaryDescription()[d].kind == particles::boundary::Kind::Periodic)
                                        throw std::runtime_error(
                                            "Boundary kind " + errorString
                                            + " is not compatible with --periodic value");
                                    T_Species::boundaryDescription()[d].kind = particles::boundary::Kind::Reflecting;
                                }
                                if(kindName == "thermal")
                                {
                                    if(T_Species::boundaryDescription()[d].kind == particles::boundary::Kind::Periodic)
                                        throw std::runtime_error(
                                            "Boundary kind " + errorString
                                            + " is not compatible with --thermal value");
                                    T_Species::boundaryDescription()[d].kind = particles::boundary::Kind::Thermal;
                                }
                                if(kindName == "periodic")
                                {
                                    // For now it must match the default-set boundary kind
                                    if(T_Species::boundaryDescription()[d].kind != particles::boundary::Kind::Periodic)
                                        throw std::runtime_error(
                                            "Boundary kind " + errorString
                                            + " is not compatible with --periodic value");
                                }
                                if(kindName == "absorbing")
                                {
                                    // For now it must match the default-set boundary kind
                                    if(T_Species::boundaryDescription()[d].kind
                                       != particles::boundary::Kind::Absorbing)
                                        throw std::runtime_error(
                                            "Boundary kind " + errorString
                                            + " is not compatible with --periodic value");
                                }
                                if((T_Species::boundaryDescription()[d].kind == particles::boundary::Kind::Periodic)
                                   && (offset != 0))
                                    throw std::runtime_error(
                                        "Periodic boundary kind " + errorString + " must have 0 boundaryOffset");
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

                    //! Boundary offsets for all axes
                    static std::vector<int32_t>& offsets()
                    {
                        static auto offsets = std::vector<int32_t>(simDim, 0);
                        return offsets;
                    }

                    //! Boundary temperatures for all axes
                    static std::vector<float_X>& temperatures()
                    {
                        static auto temperatures = std::vector<float_X>(simDim, 0.0_X);
                        return temperatures;
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
                meta::ForEach<SpeciesWithPusher, detail::ParticleBoundariesCommandLine<boost::mpl::_1>> processSpecies;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
