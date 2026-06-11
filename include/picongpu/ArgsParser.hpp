/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>

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

    protected:
        /**
         * Constructor
         */
        ArgsParser();

        ArgsParser(ArgsParser& cc);

        std::list<po::options_description> options;
    };

} // namespace picongpu
