#include "mainwindow.h"
#include <QApplication>

const std::string DEFAULT_SERVER_IP = "149.220.4.50";
const int DEFAULT_SERVER_INFOPORT = 8200;

int main(int argc, char *argv[])
{
    std::string serverip = DEFAULT_SERVER_IP;
    int server_info_port = DEFAULT_SERVER_INFOPORT;

    /// get server IP and Info Portnumber from commandline arguments
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp("--serverip", argv[i]) == 0)
        {
        serverip = argv[i];
        }
        if (strcmp("--serverinfoport", argv[i]) == 0)
        {
        serverip = argv[i];
        }
    }


    QApplication a(argc, argv);
    MainWindow w;

    w.initInfoQuery(serverip, server_info_port);

    w.show();
    //w.showMaximized();

    return a.exec();
}
