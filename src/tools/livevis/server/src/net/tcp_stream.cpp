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

#include "../../include/net/tcp_stream.hpp"
#include "../../include/net/message_ids.hpp"
#include <unistd.h>
#include <iostream>

namespace picongpu {
namespace insituvolvis {
namespace net
{

TCPStream::TCPStream(int sd, struct sockaddr_in * address)
 : m_socket(sd)
{
    char ip[50];

    ::inet_ntop(PF_INET, (struct in_addr*)&(address->sin_addr.s_addr), ip, sizeof(ip) - 1);
    m_peer_ip = ip;
    m_peer_port = ntohs(address->sin_port);

    ::sem_init(&m_sem_queue, 0, 0);
    ::pthread_create(&m_send_thread, NULL, TCPStream::send_messages, (void*)this);
}

TCPStream::~TCPStream()
{
    ::sem_destroy(&m_sem_queue);
    ::pthread_cancel(m_send_thread);

    ::close(m_socket);
}

std::string TCPStream::get_peer_ip()
{
    return m_peer_ip;
}

int TCPStream::get_peer_port()
{
    return m_peer_port;
}

void TCPStream::send(const uint32_t id, const void * data, const uint32_t length)
{
    /// is the queue full?
    int value;
    ::sem_getvalue(&m_sem_queue, &value);

    while (value > MESSAGE_QUEUE_SIZE)
    {
        /// wait for free slot in the message queue
        ::sem_getvalue(&m_sem_queue, &value);
    }

    /// put a new message in the queue
    _Message * msg = new _Message;
    msg->id = id;
    msg->length = length;
    msg->data = data;

    //std::cout << "[DEBUG] Sending ID/MsgID " << id << "/" << msg->id << " Len/MsgLen " << length << "/" << msg->length << std::endl;

    m_msg_queue.push_back(msg);
    ::sem_post(&m_sem_queue);
}

void TCPStream::receive(uint32_t * id, void *& buffer, uint32_t * length, bool wait)
{
    int bytes_recvd = 0;

    /// receive id first
    if (wait)
        bytes_recvd = ::recv(m_socket, id, sizeof(uint32_t), 0);
    else
        bytes_recvd = ::recv(m_socket, id, sizeof(uint32_t), MSG_DONTWAIT);

    if (bytes_recvd <= 0)
    {
        *id = NoMessage;
        *length = 0;
        return;
    }

    /// then length
    ::recv(m_socket, length, sizeof(uint32_t), 0);

    /// allocate buffer memory
    if (buffer == nullptr)
        buffer = new char[*length];

    /// and lastly the data (maybe in several steps)
    bytes_recvd = 0;

    while (bytes_recvd < *length)
    {
        int recvd = ::recv(m_socket, (char*)buffer + bytes_recvd, (*length) - bytes_recvd, 0);
        bytes_recvd += recvd;
    }
}

void TCPStream::wait_async_completed()
{
    int value = 1;

    while (value > 0)
        ::sem_getvalue(&m_sem_queue, &value);
}

void * TCPStream::send_messages(void * str)
{
    TCPStream * stream =  static_cast<TCPStream*>(str);

    while (1)
    {
        /// send the next message in the queue
        ::sem_wait(&(stream->m_sem_queue));

        /// get message
        _Message * msg = stream->m_msg_queue.front();

        uint32_t id = msg->id;
        uint32_t length = msg->length;
        const void * data = msg->data;

        /// send id and  length
        ::send(stream->m_socket, &id, sizeof(uint32_t), 0);
        ::send(stream->m_socket, &length, sizeof(uint32_t), 0);

        /// lastly send the data (big packages might be sent in several chunks)
        int bytes_sent = 0;
        while ( bytes_sent < length )
        {
            int sent = ::send(stream->m_socket, (char*)data + bytes_sent, length - bytes_sent, 0);
            bytes_sent += sent;
        }

        stream->m_msg_queue.pop_front();

        delete msg;
    }

    return NULL;
}

} /* end of net */
} /* end of insituvolvis*/
} /* end of picongpu */
