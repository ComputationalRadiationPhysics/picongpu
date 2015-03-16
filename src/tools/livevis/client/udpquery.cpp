#include "udpquery.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <QDebug>

#include "message_ids.hpp"

UDPQuery::UDPQuery(std::string ip, int port)
    : QObject(),
      m_ip(ip),
      m_port(port),
      m_timer(nullptr),
      m_waiting(false)
{
}

void UDPQuery::start()
{
    m_timer = new QTimer();
    m_timer->setInterval(5000);
    m_timer->setSingleShot(false);
    connect(m_timer, SIGNAL(timeout()), this, SLOT(refresh()));
    m_timer->start();
}

void UDPQuery::refresh()
{
    if (m_waiting) return;
    m_waiting = true;

    emit refreshing_list();

    /// query the visualization server connection less via UDP
    int socket_fd = ::socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in servaddr;

    ::bzero(&servaddr, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr(m_ip.c_str());
    servaddr.sin_port = htons(m_port);

    uint32_t buffer = ListVisualizations;

    /// send List request
    if ( ::sendto(socket_fd, (void*)&buffer, sizeof(uint32_t), 0, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1)
    {
        qDebug() << "Request could not be sent. Server unreachable.";
        ::close(socket_fd);
        return;
    }

    /// receive answer - count first (length of visualization list)
    uint32_t count;

    ::recvfrom(socket_fd, (void*)&count, sizeof(uint32_t), 0, NULL, NULL);

    qDebug() << "Received Count " << count;

    /// receive name and uri of all visualizations in the list
    for (int i = 0; i < count; ++i)
    {
        //char name[128];
        char name_uri[1024];

        /// receive name
        //::recvfrom(socket_fd, (void*)name, 128, 0, NULL, NULL);

        /// receive uri
        ::recvfrom(socket_fd, (void*)name_uri, 1024, 0, NULL, NULL);

        char name[128];
        char uri[1024 - 128];

        memcpy(name, name_uri, 128);
        memcpy(uri, name_uri + 128, 1024 - 128);

        memset(name + 127, 0, 1);
        memset(uri + 1024 - 128 - 1, 0, 1);

        emit received_visualization(QString(name), QString(uri));
    }

    ::close(socket_fd);

    m_waiting = false;
}
