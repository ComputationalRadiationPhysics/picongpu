/**
 * Copyright 2013-2017 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

/**
 * @mainpage PIConGPU-Frame
 *
 * Project with HZDR for porting their PiC-code to a GPU cluster.
 *
 * \image html ../../doc/logo/pic_logo_320x140.png
 *
 * @author Heiko Burau, Rene Widera, Wolfgang Hoenig, Felix Schmitt, Axel Huebl, Michael Bussmann, Guido Juckeland
 */

#include "ArgsParser.hpp"
#include "communication/manager_common.hpp"

#include <simulation_defines.hpp>
#include <mpi.h>


using namespace PMacc;
using namespace picongpu;

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    picongpu::simulation_starter::SimStarter sim;
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
            /*set error code to valid (1) after the simulation terminates*/
        case ArgsParser::SUCCESS_EXIT:
            errorCode = 0;
            break;
    };

    // Required by scorep for flushing the buffers
    cudaDeviceSynchronize();
    MPI_CHECK(MPI_Finalize());
    return errorCode;
}
