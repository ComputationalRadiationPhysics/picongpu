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
