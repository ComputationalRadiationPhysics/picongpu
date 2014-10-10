#ifndef UDPQUERY_H
#define UDPQUERY_H

#include <QObject>
#include <QTimer>

class UDPQuery : public QObject
{
    Q_OBJECT

public:

    explicit UDPQuery(std::string ip, int port);

public slots:

    void start();

private slots:

    void refresh();

signals:

    void received_visualization(QString name, QString uri);
    void refreshing_list();

private:

    QTimer * m_timer;
    bool m_waiting;

    std::string m_ip;
    int m_port;
};

#endif // UDPQUERY_H
