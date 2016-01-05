/**
 * Copyright 2013-2016 Benjamin Schneider, Rene Widera
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

#ifndef IN_SITU_NET_TCP_STREAM_HPP
#define IN_SITU_NET_TCP_STREAM_HPP

#include <string>
#include <deque>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <semaphore.h>

namespace picongpu {
namespace insituvolvis {
namespace net {

const int MESSAGE_QUEUE_SIZE = 128;

/**
 * Encapsulates a TCP/IP connection which can be used to exchange messages
 * and image data.
 */
class TCPStream
{
public:

    /** Allow connector and acceptor to instantiate objects of this class. */
    friend class TCPAcceptor;
    friend class TCPConnector;

    /**
     * Closes socket upon destruction.
     */
    ~TCPStream();

    /**
     * Send a message.
     *
     * @param id An identifier that is used to determine of which type the message is.
     * @param data The data to be sent.
     * @param length The length of the data array in bytes.
     */
    void send(const uint32_t id, const void * data, const uint32_t length);

    /**
     * Receive a message.
     *
     * @param id Output parameter to which the message ID is written.
     * @param buffer A reference to a pointer to which the received data should be written.
     * @param length Output parameter indicating the number of bytes written to the buffer.
     */
    void receive(uint32_t * id, void *& buffer, uint32_t * length, bool wait = true);

    /**
     * Wait for all asynchronous operations to finish.
     */
    void wait_async_completed();

    /**
     * Returns the IP address of the the communication partner.
     */
    std::string get_peer_ip();

    /**
     * Returns the port on which we are connected to our peer.
     */
    int get_peer_port();

private:

    /** Internal structure for messages. */
    struct _Message
    {
        uint32_t id;
        uint32_t length;
        const void * data;
    };

    /** Socket on which data is sent and received. */
    int m_socket;

    /** IP address of communication partner. */
    std::string m_peer_ip;

    /** Port */
    int m_peer_port;

    /** Semaphore to synchronize access to the message queue. */
    sem_t m_sem_queue;

    /** Worker thread which sends messages in queue. */
    pthread_t m_send_thread;

    /** Method which is executed in worker thread. */
    static void * send_messages(void * str);

    /** The message queue. */
    std::deque<TCPStream::_Message*> m_msg_queue;

    /** Disallow manual creation of objects. */
    TCPStream();
    TCPStream(const TCPStream& stream);

    /** Hidden Contructor */
    TCPStream(int sd, struct sockaddr_in * address);
};

} /* end of net */
} /* end of insituvolvis*/
} /* end of picongpu */

#endif /* IN_SITU_NET_TCP_STREAM_HPP */
