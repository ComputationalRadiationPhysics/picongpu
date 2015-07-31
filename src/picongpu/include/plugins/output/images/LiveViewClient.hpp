/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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



#ifndef LIVEVIEWCLIENT_HPP
#define    LIVEVIEWCLIENT_HPP

#include "plugins/output/sockets/SocketConnector.hpp"

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include "plugins/output/header/MessageHeader.hpp"
#include "memory/boxes/PitchedBox.hpp"
#include "memory/boxes/DataBox.hpp"


namespace picongpu
{
    using namespace PMacc;


    struct uint8_t3
    {

        uint8_t3(uint8_t x, uint8_t y, uint8_t z) : x(x), y(y), z(z)
        {
        }

        uint8_t x;
        uint8_t y;
        uint8_t z;
    };

    struct LiveViewClient
    {

        LiveViewClient(std::string ip, std::string port) : socket(NULL), ip(ip), port(port)
        {
        }

        virtual ~LiveViewClient()
        {
            __delete(socket);
        }

        template<class Box>
        void operator()(
                        const Box data,
                        const Size2D size,
                        const MessageHeader & header);

    private:
        SocketConnector *socket;
        std::string ip;
        std::string port;
    };

    template<>
    inline void LiveViewClient::operator() < DataBox<PitchedBox<float3_X, DIM2 > > >(
                                                                                   const DataBox<PitchedBox<float3_X, DIM2 > > data,
                                                                                   const Size2D size,
                                                                                   const MessageHeader& header
                                                                                   )
    {
        if (!socket)
            socket = new SocketConnector(ip, port);

        size_t elems = MessageHeader::bytes + header.window.size.productOfComponents() * sizeof (uint8_t3);
        char *array = new char[elems];

        MessageHeader * fakeHeader = (MessageHeader*) array;

        memcpy(fakeHeader, &header, sizeof(MessageHeader));

        uint8_t3 * buffer = (uint8_t3*) (array + MessageHeader::bytes);
        typedef PMacc::PitchedBox<uint8_t3, DIM2> PitchBox;
        typedef PMacc::DataBox<PitchBox > PicBox;


        PicBox smallPic(PitchBox(buffer, Size2D(), sizeof (uint8_t3) * size.x()));


        for (int y = 0; y < size.y(); ++y)
        {
            for (int x = 0; x < size.x(); ++x)
            {
                smallPic[y ][x].x = (uint8_t) (data[y ][x ].x() * 255.f);
                smallPic[y ][x].y = (uint8_t) (data[y ][x ].y() * 255.f);
                smallPic[y ][x].z = (uint8_t) (data[y ][x ].z() * 255.f);
            }
        }
        socket->send(array, elems);
        delete[] array;
    }
}

#endif    /* LIVEVIEWCLIENT_HPP */

