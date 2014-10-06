#include "../../include/net/tcp_connector.hpp"
#include <string.h>

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
