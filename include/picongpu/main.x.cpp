/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pmacc/boost_workaround.hpp>

#include "picongpu/ArgsParser.hpp"
#include "picongpu/simulation/control/SimulationStarter.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/types.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <typeinfo>

/** Run a PIConGPU simulation
 *
 * @param argc count of arguments in argv (same as for main() )
 * @param argv arguments of program start (same as for main() )
 */
int runSimulation(int argc, char** argv)
{
    using namespace picongpu;

    int errorCode = EXIT_FAILURE;
    // control the simulation lifetime
    {
        auto sim = ::picongpu::SimulationStarter{};
        auto const parserStatus = sim.parseConfigs(argc, argv);

        switch(parserStatus)
        {
        case ArgsParser::Status::error:
            errorCode = EXIT_FAILURE;
            break;
        case ArgsParser::Status::success:
            sim.load();
            sim.start();
            sim.unload();
            [[fallthrough]];
        case ArgsParser::Status::successExit:
            errorCode = 0;
            break;
        };
    }

    // finalize the pmacc context */
    pmacc::Environment<>::get().finalize();

    return errorCode;
}

/** Start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char** argv)
{
    try
    {
        return runSimulation(argc, argv);
    }
    // A last-ditch effort to report exceptions to a user
    catch(std::exception const& ex)
    {
        auto const typeName = std::string(typeid(ex).name());
        std::cerr << "Unhandled exception of type '" + typeName + "' with message '" + ex.what() + "', terminating\n";
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cerr << "Unhandled exception of unknown type, terminating\n";
        return EXIT_FAILURE;
    }
}
