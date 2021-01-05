/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera,
 *                     Benjamin Worpitz
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

#include <boost/program_options/options_description.hpp>

#include <string>
#include <stdexcept>
#include <vector>
#include <stdint.h>
#include <list>

namespace picongpu
{
    namespace po = boost::program_options;

    /**
     * Parses configuration arguments from command line and/or a configuration file.
     * Call init() before usage.
     * Implemented as Singleton.
     */
    class ArgsParser
    {
    public:
        //! Parsing status
        enum Status
        {
            success,
            successExit,
            error
        };

        /**
         * Returns an instance of ArgsParser
         *
         * @return an instance
         */
        static ArgsParser& getInstance();

        void addOptions(po::options_description desc)
        {
            options.push_back(desc);
        }

        /**
         * Parses arguments from command line and optional configuration files.
         *
         * @param argc number of command line arguments
         * @param argv command line arguments
         * @return parsing status
         */
        Status parse(int argc, char** argv);

    private:
        /**
         * Constructor
         */
        ArgsParser();

        ArgsParser(ArgsParser& cc);

        std::list<po::options_description> options;
    };

} // namespace picongpu
