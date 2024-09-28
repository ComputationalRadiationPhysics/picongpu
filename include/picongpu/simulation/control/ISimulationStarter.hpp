/* Copyright 2013-2023 Rene Widera
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

#include "picongpu/ArgsParser.hpp"
#include "picongpu/defines.hpp"

#include <pmacc/pluginSystem/IPlugin.hpp>

namespace picongpu
{
    using namespace pmacc;


    class ISimulationStarter : public IPlugin
    {
    public:
        ~ISimulationStarter() override = default;
        /**Pars progarm parameters
         *             *
         * @param argc number of arguments in argv
         * @param argv arguments for programm
         *
         * @return true if no error else false
         */
        virtual ArgsParser::Status parseConfigs(int argc, char** argv) = 0;

        /*start simulation
         * is called after parsConfig and pluginLoad
         */
        virtual void start() = 0;

        void restart(uint32_t, const std::string) override
        {
            // nothing to do here
        }

        void checkpoint(uint32_t, const std::string) override
        {
            // nothing to do here
        }
    };
} // namespace picongpu
