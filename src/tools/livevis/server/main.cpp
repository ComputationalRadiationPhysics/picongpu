/**
 * Copyright 2013-2016 Benjamin Schneider
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


#include "include/VisualizationServer.h"


const int VISSERVER_DEFEULT_PORT = 8100;
const int VISSERVER_DEFAULT_INFO_PORT = 8200;

/**
 * Main
 */
int main(int argc, char ** argv)
{
    int port = VISSERVER_DEFEULT_PORT;
    int info_port = VISSERVER_DEFAULT_INFO_PORT;

    if (argc == 2)
    {
        if ( strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 )
        {
            /// list help
            std::cout << "In Situ Visualization Server Help" << std::endl;
            std::cout << "--port [PORT] Server Port for Simulations to connect to." << std::endl;
               std::cout << "--infoport [PORT] Server Info Port where Clients can query a list of available simulations." << std::endl;

            return 0;
        }
    }

    /// read command line arguments
    for (int arg = 0; arg < argc; arg++)
    {
        if ( (strcmp(argv[arg], "--port") == 0) && (argc >= (arg + 1)) ) ::sscanf(argv[arg + 1], "%d", &port);
        if ( (strcmp(argv[arg], "--infoport") == 0) && (argc >= (arg + 1)) ) ::sscanf(argv[arg + 1], "%d", &info_port);
    }

    InSituVisualization::VisualizationServer vis_srv(port, info_port);
    int retval = vis_srv.run();

    return retval;
}
