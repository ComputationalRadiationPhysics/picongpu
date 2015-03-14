/**
 * Copyright 2013-2015 Benjamin Schneider, Axel Huebl
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

    /// get server IP and Info Portnumber from commandline arguments
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp("--serverip", argv[i]) == 0)
        {
        serverip = argv[i];
        }
        if (strcmp("--serverinfoport", argv[i]) == 0)
        {
        serverip = argv[i];
        }
    }


    QApplication a(argc, argv);
    MainWindow w;

    w.initInfoQuery(serverip, server_info_port);

    w.show();
    //w.showMaximized();

    return a.exec();
}
