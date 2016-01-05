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

#include "../include/VisualizationServer.h"
#include "../include/net/message_ids.hpp"
#include <sys/fcntl.h>

using namespace InSituVisualization;

/**
 * Reaper procedure for zombie processes.
 */
void sigchld_handler(int s)
{
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

/**
 * VisualizationServer::VisualizationServer
 */
VisualizationServer::VisualizationServer(int vis_port, int info_port)
    : m_listeningPort(vis_port),
      m_infoPort(info_port),
      m_vis_acceptor(nullptr),
      //m_info_acceptor(nullptr),
      m_rivcore(nullptr),
      m_ip_comm(nullptr)
{ }

/**
 * VisualizationServer::~VisualizationServer
 */
VisualizationServer::~VisualizationServer()
{ }

/**
 * Start server.
 */
int VisualizationServer::run()
{
    std::cout << "[SERVER] Starting the Visualization Server..." << std::endl;

    /// initialize RIVLib Core
    this->m_rivcore = rivlib::core::create();
    if (!m_rivcore)
    {
        stopListening();
        return 3;
    }

    /// create IP Communicator
    m_ip_comm = rivlib::ip_communicator::create(rivlib::ip_communicator::DEFAULT_PORT);
    this->m_rivcore->add_communicator(m_ip_comm);

    /// to kill zombies
    struct sigaction sa;
    sa.sa_handler = ::sigchld_handler;
    ::sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    if (::sigaction(SIGCHLD, &sa, NULL) == -1)
        ::exit(1);

    /// create an acceptor waiting for incoming connections
    m_vis_acceptor = new TCPAcceptor(m_listeningPort);
    //m_info_acceptor = new TCPAcceptor(m_infoPort);

    /// now server listens for connection attempts
    this->startListening();

    /// we're done, stop listening
    this->stopListening();

    return 0;
}

/**
 * Listen on the specified port for incoming connections.
 */
void VisualizationServer::startListening()
{
    m_vis_acceptor->start();
    //m_info_acceptor->start();

    /// start thread to wait for connections
    ::pthread_create(&m_listeningThread, NULL, VisualizationServer::waitForConnection, (void*)this);

    /// start thread to answer info requests
    ::pthread_create(&m_infoThread, NULL, VisualizationServer::runInfoChannel, (void*)this);

    /// wait for user input
    while(1)
    {
        std::cout << "[SERVER] List visualizations(l), server info(i) or Quit(q): ";
        char in;

        std::cin >> in;

        if (in == 'l')
        {
            /// list connected visualizations
            std::cout << "\n=========================" << std::endl;
            std::cout << "Connected Visualizations:" << std::endl;
            std::cout << "=========================" << std::endl;
            std::cout << "Name              Status  URI" << std::endl;
            std::cout << "----------------  ------  ---------------------------" << std::endl;
            for (std::vector<Visualization*>::const_iterator it = m_visualizations.begin(); it != m_visualizations.end(); it++)
            {
                /// get vis name
                std::string name = (*it)->getName();

                /// get uri
                size_t urilen = this->m_ip_comm->public_uri( (*it)->getProvider(), 0, nullptr, 0 );
                char * c = new char[urilen + 1];
                c[urilen] = 0;
                size_t len2 = this->m_ip_comm->public_uri( (*it)->getProvider(), 0, c, urilen );
                c[len2] = 0;
                std::string uri(c);
                delete [] c;

                /// get status
                std::string status = (*it)->isRunning() ? "R" : "S";

                /// print info line
                std::cout << std::left << std::setw(18) << name << std::setw(8) << status << std::setw(64) << uri << std::endl;
            }
            std::cout << "-----------------------------------------------------\n" << std::endl;
        }
        else if (in == 'i')
        {
            std::cout << "\n===================================" << std::endl;
            std::cout << "Network Interfaces:" << std::endl;
            std::cout << "===================================" << std::endl;
            showServerIPs();
            std::cout << std::endl;
            std::cout << "\n[SERVER] Waiting for connections on Port: " << m_listeningPort << std::endl;
            std::cout << "[SERVER] Answering information requests on Port: " << m_infoPort << "\n" << std::endl;
        }
        else if (in == 'q')
        {
            break;
        }
    }
}

/**
 * Method to run the info channel thread, answering requests (e.g. to list the available visualizations).
 */
void * VisualizationServer::runInfoChannel(void * server)
{
    VisualizationServer * srv = static_cast<VisualizationServer*>(server);

    //uint32_t count = 0;
    std::string name, uri;

    uint32_t id;

    /// create socket
    srv->m_info_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (srv->m_info_socket < 0)
    {
        ::perror("Cannot create UDP Socket!");
        ::exit(1);
    }

    struct sockaddr_in myaddr;

    memset((char*)&myaddr, 0, sizeof(myaddr));
    myaddr.sin_family = AF_INET;
    myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    myaddr.sin_port = htons(srv->m_infoPort);

    /// bind it to an address/port
    if ( ::bind(srv->m_info_socket, (struct sockaddr*)&myaddr, sizeof(myaddr)) < 0)
    {
        ::perror("Failed to bind UDP Socket!");
        ::exit(1);
    }

    while(1)
    {
        struct sockaddr_in remote_addr;
        socklen_t addrlen = sizeof(remote_addr);

        ::recvfrom(srv->m_info_socket, (void*)&id, sizeof(uint32_t), 0, (struct sockaddr*)&remote_addr, &addrlen);

        std::cout << "[INFO CHANNEL] Received ID " << id << std::endl;

        if (id == ListVisualizations)
        {
            /// respond with a list of available visualizations
            /// first, send length of the list
            uint32_t count = srv->m_visualizations.size();

            std::cout << "[INFO CHANNEL] Sending Count " << count << std::endl;

            ::sendto(srv->m_info_socket, (void*)&count, sizeof(uint32_t), 0, (struct sockaddr*)&remote_addr, sizeof(remote_addr));

            /// send a list of the available visualizations
            for (std::vector<Visualization*>::iterator it = srv->m_visualizations.begin(); it != srv->m_visualizations.end(); it++)
            {
                /// get vis name and send it
                char name[128];
                memset(name, 0, 128);

                std::string namestr = (*it)->getName();
                int namelen = strlen(namestr.c_str());

                memcpy(name, namestr.c_str(), namelen);
                name[namelen] = '\0';

                // ::sendto(srv->m_info_socket, name, 128, 0, (struct sockaddr*)&remote_addr, sizeof(remote_addr));

                /// get uri and send it
                size_t urilen = srv->m_ip_comm->public_uri( (*it)->getProvider(), 0, nullptr, 0 );
                char * c = new char[urilen + 1];
                c[urilen] = 0;
                size_t len2 = srv->m_ip_comm->public_uri( (*it)->getProvider(), 0, c, urilen );
                c[len2] = 0;
                std::string uristr = std::string(c);
                delete [] c;

                char uri[1024 - 128];
                memset(uri, 0, 1024 - 128);

                int uristrlen = strlen(uristr.c_str());

                memcpy(uri, uristr.c_str(), uristrlen);
                uri[uristrlen] = '\0';

                /// combine name and uri
                char name_uri[1024];
                memset(name_uri, 0, 1024);

                memcpy(name_uri, name, 128);
                memcpy(name_uri + 128, uri, 1024 - 128);

                ::sendto(srv->m_info_socket, name_uri, 1024, 0, (struct sockaddr*)&remote_addr, sizeof(remote_addr));

                // ::sendto(srv->m_info_socket, uri, 1024, 0, (struct sockaddr*)&remote_addr, sizeof(remote_addr));

                std::cout << "Vis: " << namestr << " " << uristr << std::endl;
                std::cout << "Combined: " << name_uri << std::endl;
            }
        }
    }

    return nullptr;
}

/**
 * Wait for a simulation to connect.
 */
void * VisualizationServer::waitForConnection(void * server)
{
    VisualizationServer * srv = static_cast<VisualizationServer*>(server);

    while(1)
    {
        srv->acceptConnection();
    }

    return nullptr;
}

/**
 * Accept a connection attempt and create a new Visualization instance.
 */
void VisualizationServer::acceptConnection()
{
    TCPStream * new_stream = m_vis_acceptor->accept();

    if (new_stream == NULL)
    {
        std::cout << "[SERVER] Connection attempt failed!" << std::endl;
    }
    else
    {
        std::cout << "\n[SERVER] Incoming connection from " << new_stream->get_peer_ip() << std::endl;

        /// receive name of simulation on newly connected socket
        uint32_t id = 0;
        uint32_t len = 0;
        void * name_buffer = nullptr;

        new_stream->receive(&id, name_buffer, &len);

        if (id != MessageID::VisName)
        {
            std::cerr << "Error: Expected VisName message ID! Received ID " << id << " Length " << len << std::endl;
            return;
        }

        char visName[len + 1];
        memcpy(visName, name_buffer, len);

        visName[len] = '\0';

        std::string new_name(visName);

        std::cout << "Newly connected Vis: " << new_name << "." << std::endl;

        delete [] (char*)name_buffer;

        /// receive image resolution
        void * is_buffer = nullptr;
        uint32_t is_id;
        uint32_t is_len;

        new_stream->receive(&is_id, is_buffer, &is_len);

        if (is_id != MessageID::ImageSize)
        {
            std::cerr << "Error: Expected ImageSize message ID!" << std::endl;
            return;
        }

        uint32_t new_width = reinterpret_cast<uint32_t*>(is_buffer)[0];
        uint32_t new_height = reinterpret_cast<uint32_t*>(is_buffer)[1];

        /// a new visualization connected, create a Visualization instance to handle it
        this->startVisAsync(new_stream, new_name, new_width, new_height);

        delete [] (char*)is_buffer;
    }
}

/**
 * Start a new thread running the communication among server and simulation.
 */
void VisualizationServer::startVisAsync(TCPStream * stream, std::string name, int width, int height)
{
    /// create a RIV provider for the newly connected visualization
    rivlib::provider::ptr prov(rivlib::provider::create_provider(name.c_str()));

    this->m_rivcore->add_provider(prov);

    /// create a new visualization object
    Visualization * vis = new Visualization(stream, prov, name, width, height);

    m_visualizations.push_back(vis);

    /// run the visualization in a seperate thread
    vis->runAsync();
}

/**
 * Helper method to get IP address.
 */
void* VisualizationServer::get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET)
    {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }
    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

/**
 * Shows IP addresses on which the server can be reached.
 */
void VisualizationServer::showServerIPs()
{
    struct ifaddrs *ifaddr, *ifa;
    int family, s;
    char host[NI_MAXHOST];

    if (::getifaddrs(&ifaddr) == -1)
    {
        ::perror("getifaddrs");
        ::exit(EXIT_FAILURE);
    }

    /// Walk through linked list, maintaining head pointer so we
    /// can free list later
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;

        /// Display interface name and family (including symbolic
        /// form of the latter for the common families)
        ::printf("%s  address family: %d%s\n",
               ifa->ifa_name, family,
               (family == AF_PACKET) ? " (AF_PACKET)" :
               (family == AF_INET) ?   " (AF_INET)" :
               (family == AF_INET6) ?  " (AF_INET6)" : "");

        /// For an AF_INET* interface address, display the address
        if (family == AF_INET || family == AF_INET6)
        {
            s = ::getnameinfo(ifa->ifa_addr,
                            (family == AF_INET) ? sizeof(struct sockaddr_in) :
                            sizeof(struct sockaddr_in6),
                            host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);

            if (s != 0)
            {
                ::printf("getnameinfo() failed: %s\n", ::gai_strerror(s));
                ::exit(EXIT_FAILURE);
            }
            ::printf("\taddress: <%s>\n", host);
        }
    }

    ::freeifaddrs(ifaddr);
}

/**
 * Stop the server and do not accept incoming connections anymore.
 */
void VisualizationServer::stopListening()
{
    /// kill info and vis connection threads
    ::pthread_cancel(m_listeningThread);
    ::pthread_cancel(m_infoThread);

    /// close acceptor
    delete m_vis_acceptor;
    //delete m_info_acceptor;
    ::close(m_info_socket);

    /// kill all visualization threads
    for (std::vector<Visualization*>::iterator it = m_visualizations.begin(); it != m_visualizations.end(); it++)
    {
        delete *it;
    }

    m_visualizations.clear();

    /// shutdown RIVLib
    this->m_rivcore->shutdown();
    this->m_rivcore.reset();
}
