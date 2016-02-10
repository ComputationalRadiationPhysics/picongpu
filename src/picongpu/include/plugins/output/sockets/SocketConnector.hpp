/**
 * Copyright 2013-2016 Rene Widera, Axel Huebl
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

#include "pmacc_types.hpp"
#include "plugins/output/header/MessageHeader.hpp"

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

#include <iostream>

#include "plugins/output/compression/ZipConnector.hpp"
#include <sstream>

namespace picongpu
{

class SocketConnector
{
private:

    /** convert little endian 16 Bit unsigned integer to big endian
     * This function is used instead htons which create warnings at compile time
     * "warning: "cc" clobber ignored"
     */
    uint16_t littleToBigEndian(uint16_t value)
    {
        return ((value & 0xFF00) >> 8)+
            ((value & 0x00FF) << 8);
    }
public:

    SocketConnector(std::string ip, std::string port) : connectOK(true)
    {
        SocketFD = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        if (-1 == SocketFD)
        {
            perror("cannot create socket");
            exit(EXIT_FAILURE);
        }

        memset(&stSockAddr, 0, sizeof (stSockAddr));

        stSockAddr.sin_family = AF_INET;
        std::stringstream s_port(port);
        uint16_t portAsInt;
        s_port >> portAsInt;
        stSockAddr.sin_port = littleToBigEndian(portAsInt);

        Res = inet_pton(AF_INET, ip.c_str(), &stSockAddr.sin_addr);

        if (0 > Res)
        {
            perror("error: first parameter is not a valid address family");
            close(SocketFD);
            connectOK = false;
        }
        else if (0 == Res)
        {
            perror("char string (second parameter does not contain valid ipaddress)");
            close(SocketFD);
            connectOK = false;
        }

        if (-1 == connect(SocketFD, (struct sockaddr *) &stSockAddr, sizeof (stSockAddr)))
        {

            perror("connect failed");
            close(SocketFD);
            connectOK = false;
        }

    }

    void send(void* array, size_t size)
    {
        if (connectOK)
        {
            char* tmp = new char[size];
            memcpy(tmp, array, sizeof(MessageHeader));

            ZipConnector zip;
            size_t zipedSize = zip.compress(tmp + MessageHeader::bytes, ((char*) array) + MessageHeader::bytes, size - MessageHeader::bytes, 6);
            MessageHeader* header = (MessageHeader*) tmp;
            header->data.byte = (uint32_t) zipedSize;
            write(SocketFD, tmp, zipedSize + MessageHeader::bytes);
            __deleteArray(tmp);
        }
    }

    virtual ~SocketConnector()
    {
        if (connectOK)
        {
            shutdown(SocketFD, SHUT_RDWR);
            close(SocketFD);
        }
    }

private:
    struct sockaddr_in stSockAddr;
    int Res;
    int SocketFD;
    bool connectOK;

};

}


