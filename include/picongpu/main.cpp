/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "picongpu/ArgsParser.hpp"
#include <pmacc/Environment.hpp>
#include <pmacc/types.hpp>

#include <picongpu/simulation_defines.hpp>


/* Workaround for Visual Studio to avoid a collision between ERROR macro
 * defined in wingdi.h file (included from some standard library headers) and
 * enumerator ArgsParser::ArgsErrorCode::ERROR.
 */
#ifdef _MSC_VER
#   undef ERROR
#endif

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char **argv)
{
    using namespace picongpu;

    simulation_starter::SimStarter sim;
    ArgsParser::ArgsErrorCode parserCode = sim.parseConfigs(argc, argv);
    int errorCode = 1;

    switch(parserCode)
    {
        case ArgsParser::ERROR:
            errorCode = 1;
            break;
        case ArgsParser::SUCCESS:
            sim.load();
            sim.start();
            sim.unload();
            PMACC_FALLTHROUGH;
        case ArgsParser::SUCCESS_EXIT:
            errorCode = 0;
            break;
    };

    /* finalize the pmacc context */
    pmacc::Environment<>::get().finalize();

    return errorCode;
}
