/**
 * Copyright 2013-2016 Benjamin Schneider, Axel Huebl
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

#include "../../include/net/tcp_connector.hpp"
#include <cstring>

namespace picongpu {
namespace insituvolvis {
namespace net
{

TCPStream * TCPConnector::connect(std::string ip, int port)
{
    struct sockaddr_in address;

    ::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(port);

    if (resolve_hostname(ip.c_str(), &(address.sin_addr)) != 0)
    {
        ::inet_pton(PF_INET, ip.c_str(), &(address.sin_addr));
    }

    int sd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (::connect(sd, (struct sockaddr*)&address, sizeof(address)) != 0)
    {
        return NULL;
    }

    return new TCPStream(sd, &address);
}

int TCPConnector::resolve_hostname(const char * hostname, struct in_addr * addr)
{
    struct addrinfo * res;

    int result = ::getaddrinfo(hostname, NULL, NULL, &res);
    if (result == 0)
    {
        ::memcpy(addr, &((struct sockaddr_in *)res->ai_addr)->sin_addr, sizeof(struct in_addr));
        ::freeaddrinfo(res);
    }
    return result;
}

} /* end of net */
} /* end of insituvolvis*/
} /* end of picongpu */
