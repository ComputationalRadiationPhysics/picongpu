#include "../../include/net/tcp_acceptor.hpp"
#include <unistd.h>
#include <string.h>
#include <iostream>

namespace picongpu {
namespace insituvolvis {
namespace net
{

TCPAcceptor::TCPAcceptor(int port, const char * address)
 : m_listening_socket(0),
   m_port(port),
   m_address(address),
   m_is_listening(false)
{ }

TCPAcceptor::~TCPAcceptor()
{
    if (m_listening_socket > 0)
        ::close(m_listening_socket);
}

int TCPAcceptor::start()
{
    if (m_is_listening)
        return 0;

    m_listening_socket = ::socket(PF_INET, SOCK_STREAM, 0);

    struct sockaddr_in address;
    ::memset(&address, 0, sizeof(address));
    address.sin_family = PF_INET;
    address.sin_port = htons(m_port);

    if (m_address.size() > 0)
    {
        ::inet_pton(PF_INET, m_address.c_str(), &(address.sin_addr));
    }
    else
    {
        address.sin_addr.s_addr = INADDR_ANY;
    }

    int result = ::bind(m_listening_socket, (struct sockaddr*)&address, sizeof(address));
    if (result != 0)
    {
        std::cerr << "Not able to bind socket!" << std::endl;
        return result;
    }

    result = ::listen(m_listening_socket, 10);

    if (result != 0)
    {
        std::cerr << "Not able to listen!" << std::endl;
        return result;
    }

    m_is_listening = true;
    return result;
}

TCPStream * TCPAcceptor::accept()
{
    if (!m_is_listening)
        return NULL;

    struct sockaddr_in address;
    socklen_t len = sizeof(address);
    ::memset(&address, 0, sizeof(address));

    int sd = ::accept(m_listening_socket, (struct sockaddr*)&address, &len);

    if (sd < 0)
    {
        std::cerr << "Accept failed!" << std::endl;
        return NULL;
    }

    return new TCPStream(sd, &address);
}

} /* end of net */
} /* end of insituvolvis*/
} /* end of picongpu */
