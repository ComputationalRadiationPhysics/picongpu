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

#ifndef INSITU_NET_TCP_CONNECTOR_HPP
#define INSITU_NET_TCP_CONNECTOR_HPP

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

#include "tcp_stream.hpp"

namespace picongpu {
namespace insituvolvis {
namespace net
{

/**
 * Actively establishes a connection to a server via connect().
 */
class TCPConnector
{
public:

    /**
     * Try to connect to server.
     *
     * @param ip IP address of the server we want to connect to.
     * @param port Port on which we try to connect.
     */
    TCPStream * connect(std::string ip, int port);

private:

    int resolve_hostname(const char * hostname, struct in_addr * addr);
};

} /* end of net */
} /* end of insituvolvis*/
} /* end of picongpu */

#endif /* INSITU_NET_TCP_CONNECTOR_HPP */
