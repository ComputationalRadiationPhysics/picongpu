/**
 * Copyright 2013-2016 Benjamin Schneider, Axel Huebl, Richard Pausch
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

#include "mainwindow.h"
#include <QApplication>

const std::string DEFAULT_SERVER_IP = "127.0.0.1";//"149.220.4.37";
const int DEFAULT_SERVER_INFOPORT = 8200;

int main(int argc, char *argv[])
{
    std::string serverip = DEFAULT_SERVER_IP;
    int server_info_port = DEFAULT_SERVER_INFOPORT;

    /// get server IP and Info Portnumber from command line arguments
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp("--serverip", argv[i]) == 0 && i < argc-1)
        {
            serverip = argv[i+1];
        }
        if (strcmp("--serverinfoport", argv[i]) == 0 && i < argc-1)
        {
            server_info_port = atoi(argv[i+1]);
        }
        if ((strcmp("--help", argv[i]) == 0) or (strcmp("-h", argv[i]) == 0))
        {
            printf("This is the client for the live visualization tool of PIConGPU.\n\n");
            printf("--serverip       : Sets the IP address of the visualization server. [Default: %s]\n", serverip.c_str());
            printf("--serverinfoport : Sets the port on which to connect to the visualization server. [Default: %d]\n", server_info_port);
            printf("--help , -h      : Show usage information.\n");
            return 0;
        }
    }


    QApplication a(argc, argv);
    MainWindow w;

    w.initInfoQuery(serverip, server_info_port);

    w.show();
    //w.showMaximized();

    return a.exec();
}
