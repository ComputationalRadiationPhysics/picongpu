/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/ArgsParser.hpp"

#include "picongpu/debug/PIConGPUVerbose.hpp"
#include "picongpu/versionFormat.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace picongpu
{
    namespace
    {
        /** Report deprecated parameters
         *
         * This function is meant to handle cases when some parameters are changed
         * but the old versions temporarily kept for backward compatibility and
         * deprecated. Notably, this applies to compile-time parameters getting a
         * run-time version. Hence it deliberately ignores incapsulation and code
         * duplication and simply has a hardcoded set of cases.
         */
        void reportDeprecated(boost::program_options::variables_map const& vm)
        {
            using pmacc::log;
            using Level = PIConGPUVerbose::PHYSICS;
        }

    } // anonymous namespace

    ArgsParser::ArgsParser() = default;

    ArgsParser::ArgsParser(ArgsParser&)
    {
    }

    template<class T>
    bool from_string(T& t, std::string const& s, std::ios_base& (*f)(std::ios_base&) )
    {
        std::istringstream iss(s);
        if((iss >> f >> t).fail())
            throw std::invalid_argument("convertion invalid!");

        return true;
    }

    ArgsParser& ArgsParser::getInstance()
    {
        static ArgsParser instance;
        return instance;
    }

    ArgsParser::Status ArgsParser::parse(int argc, char** argv)
    {
        namespace po = boost::program_options;

        try
        {
            // add help message
            std::stringstream desc_stream;
            desc_stream << "Usage picongpu [-d dx=1 dy=1 dz=1] -g width height depth [options]" << std::endl;

            po::options_description desc(desc_stream.str());

            std::vector<std::string> config_files;

            // add possible options
            desc.add_options()("help,h", "print help message and exit")(
                "validate",
                "validate command line parameters and exit")("version,v", "print version information and exit")(
                "config,c",
                po::value<std::vector<std::string>>(&config_files)->multitoken(),
                "Config file(s)");

            // add all options from plugins
            for(auto iter = options.begin(); iter != options.end(); ++iter)
                desc.add(*iter);

            // parse command line options and config file and store values in vm
            po::variables_map vm;
            // log<picLog::SIMULATION_STATE > ("parsing command line");
            po::store(po::parse_command_line(argc, argv, desc), vm);

            if(vm.count("config"))
            {
                std::vector<std::string> conf_files = vm["config"].as<std::vector<std::string>>();

                for(auto iter = conf_files.begin(); iter != conf_files.end(); ++iter)
                {
                    // log<picLog::SIMULATION_STATE > ("parsing config file '%1%'") % (*iter);
                    std::ifstream config_file_stream(iter->c_str());
                    po::store(po::parse_config_file(config_file_stream, desc), vm);
                }
            }

            po::notify(vm);

            // print help message and quit simulation
            if(vm.count("help"))
            {
                std::cout << desc << "\n";
                return Status::successExit;
            }
            // print versions of dependent software
            if(vm.count("version"))
            {
                void(getSoftwareVersions(std::cout));
                return Status::successExit;
            }
            // no parameters set: required parameters (e.g., -g) will be missing
            // -> obvious wrong usage
            // -> print help and exit with error code
            if(argc == 1) // argc[0] is always the program name
            {
                std::cerr << desc << "\n";
                return Status::error;
            }

            reportDeprecated(vm);

            if(vm.count("validate"))
            {
                /* if we reach this part of code the parameters are valid
                 * and the option `validate` is set.
                 */
                return Status::successExit;
            }
        }
        catch(po::error const& e)
        {
            std::cerr << e.what() << std::endl;
            return Status::error;
        }

        return Status::success;
    }

} // namespace picongpu
