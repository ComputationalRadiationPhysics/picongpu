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

#ifndef VISUALIZATIONSERVER_H
#define VISUALIZATIONSERVER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#include <cstring>
#include <unistd.h>
#include <cerrno>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <csignal>
#include <ifaddrs.h>
#include <pthread.h>

#include <rivlib/rivlib.h>

#include "../include/Visualization.h"
#include "../include/net/tcp_acceptor.hpp"

namespace InSituVisualization
{

using namespace eu_vicci;
using namespace picongpu::insituvolvis::net;

const int DEFAULT_INFO_PORT = 8200;

/** reaper method for zombie processes */
void sigchld_handler(int s);

/**
 * A Server to accept TCP connections from visualizations that want to send their image output and receive
 * control commands (e.g. camera position). For every connected visualization a RIVLib provider is created
 * to communicate with viewer clients.
 */
class VisualizationServer
{
public:

    explicit VisualizationServer(int vis_port, int info_port = DEFAULT_INFO_PORT);
    virtual ~VisualizationServer();

    /**
     * Start the visualization server.
     */
    int run();

protected:

    /**
     * Starts listening for incoming connections on the
     * constructor-specified port in a separate thread.
     */
    void startListening();

    /** Closes the listening socket and thread. */
    void stopListening();

    /**
     * Creates a new Visualization object and starts it. Internally
     * the visualization runs in its own thread.
     */
    void startVisAsync(TCPStream * stream, std::string name, int width, int height);

    /**
     * Helper methods to show IP of the host running this server
     * so visualizations know to which IP to connect.
     */
    void showServerIPs();
    void* get_in_addr(struct sockaddr *sa);

    /** Wrapper method for starting the thread on the server. */
    static void * waitForConnection(void * server);

    /** Here the server is waiting for a new connection to come in. */
    void acceptConnection();

    /** Run a thread to answer info queries (e.g. deliver a list of connected visualizations). */
    static void * runInfoChannel(void * server);

private:

    /** Port on which the server is listening for connections from visualizations. */
    int m_listeningPort;

    int m_infoPort;

    /** TCPAcceptor to accept incoming connections. */
    TCPAcceptor * m_vis_acceptor;

    //TCPAcceptor * m_info_acceptor;
    int m_info_socket;

    /** The thread in which the server accepts connection attempts. */
    pthread_t m_listeningThread;

    /** The thread running the info channel. */
    pthread_t m_infoThread;

    /** The RIVLib core. */
    rivlib::core::ptr m_rivcore;

    /** The RIVLib ip communicator. */
    rivlib::ip_communicator::ptr m_ip_comm;

    /** List of currently connected visualizations. */
    std::vector<Visualization*> m_visualizations;
};

} /* end of namespace InSituVisualization */

#endif // VISUALIZATIONSERVER_H
