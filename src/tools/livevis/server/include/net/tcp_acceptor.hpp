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

#ifndef INSITU_NET_TCP_ACCEPTOR_HPP
#define INSITU_NET_TCP_ACCEPTOR_HPP

#include "tcp_stream.hpp"

namespace picongpu {
namespace insituvolvis {
namespace net
{

/**
 * Waits for connection attempts and returns a TCPStream if a connection
 * was established.
 */
class TCPAcceptor
{
public:

    /**
     * Create a new acceptor.
     *
     * @param port Port to listen on for incoming connections.
     * @param address Address to listen on (optional).
     */
    TCPAcceptor(int port, const char * address = "");

    /**
     * Stops listening when destructed.
     */
    ~TCPAcceptor();

    /**
     * Start listening for connection attempts.
     *
     * @return An error code or zero on success.
     */
    int start();

    /**
     * Accept a new connection.
     */
    TCPStream * accept();

private:

    TCPAcceptor() {}

    /** Socket which listens. */
    int m_listening_socket;

    /** Port on which acceptor listens. */
    int m_port;

    /** Address of the listener. */
    std::string m_address;

    /** Are we already listening. */
    bool m_is_listening;
};

} /* end of net */
} /* end of insituvolvis*/
} /* end of picongpu */

#endif /* INSITU_NET_TCP_ACCEPTOR_HPP */
