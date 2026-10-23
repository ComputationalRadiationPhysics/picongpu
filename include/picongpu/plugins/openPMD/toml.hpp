/* Copyright 2021-2024 Franz Poeschel
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

#include "picongpu/plugins/openPMD/Parameters.hpp"

#include <any>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mpi.h>

namespace picongpu
{
    namespace toml
    {
        /* TOML parameters split into interface ITomlParameter and implementation TomlParameter<TargetType> because a
         * parameter might have different types (string, int, boolean flags, ...). */
        struct ITomlParameter
        {
            std::string optionName;

            ITomlParameter(std::string optionName_in) : optionName(std::move(optionName_in))
            {
            }

            virtual ~ITomlParameter() = default;

            virtual void parseOption(std::any tomlConfig, openPMD::PluginParameters& options) const = 0;
        };

        template<typename TargetType>
        struct TomlParameter : ITomlParameter
        {
            TargetType openPMD::PluginParameters::* destination;

            TomlParameter(std::string optionName_in, TargetType openPMD::PluginParameters::* destination_in)
                : ITomlParameter{std::move(optionName_in)}
                , destination(destination_in)
            {
            }

            void parseOption(std::any tomlConfig, openPMD::PluginParameters& options) const override;
        };

        template<typename TargetType, typename... Args>
        auto makeTomlParameter(Args&&... args) -> std::unique_ptr<ITomlParameter>
        {
            return std::unique_ptr<ITomlParameter>(new TomlParameter<TargetType>{std::forward<Args>(args)...});
        }

        // We can't use pmacc::pluginSystem::Slice in a hostonly file due to PIConGPU include structure
        // so reimplement it here
        struct TimeSlice
        {
            uint32_t start = 0;
            // plz leave this like this, uint32_t end = -1 is giving me segfaults
            // (probably because this is included in host-side and device-side code)
            uint32_t end = std::numeric_limits<uint32_t>::max();
            uint32_t period = 1;

            std::string asString() const;
        };

        struct Periodicity
        {
            TimeSlice timeSlice;
            std::vector<std::string> sources;
        };

        class DataSources
        {
        public:
            using SimulationStep_t = uint32_t;

        private:
            std::vector<Periodicity> m_periods;

        public:
            openPMD::PluginParameters openPMDPluginParameters;

            DataSources(
                std::string const& tomlFile,
                std::vector<std::unique_ptr<picongpu::toml::ITomlParameter>> tomlParameters,
                std::vector<std::string> const& allowedDataSources,
                MPI_Comm comm,
                openPMD::PluginParameters pluginParameters);

            /*
             * The datasources that are active at currentStep().
             */
            std::vector<std::string> currentDataSources(SimulationStep_t step) const;

            /*
             * Return a string that can be used for
             * Environment<>::get().PluginConnector().setNotificationPeriod().
             */
            std::string periods() const;
        };

        // Definitions of these need to go in NVCC-compiled files
        // (openPMDWriter.hpp) due to include structure of PIConGPU
        void writeLog(char const* message, size_t argsn = 0, char const* const* argsv = nullptr);
        std::vector<TimeSlice> parseTimeSlice(std::string const&);
    } // namespace toml
} // namespace picongpu
