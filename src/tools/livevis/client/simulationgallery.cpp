#include "simulationgallery.h"
#include "ui_simulationgallery.h"

#include <QDebug>
#include <QImage>
#include <QPainter>

#include "udpquery.h"


SimulationGallery::SimulationGallery(QWidget *parent) :
    PictureFlow(parent),
    ui(new Ui::SimulationGallery),
    m_num_slides(0)
{
    ui->setupUi(this);
}

SimulationGallery::~SimulationGallery()
{
    delete m_receiver_thread;
    delete ui;
}

void SimulationGallery::startQuery(std::string ip, int port)
{
    m_visserver_ip = ip;
    m_visserver_infoport = port;

    /// create thread not to block GUI
    m_receiver_thread = new QThread();

    /// create query object
    UDPQuery * query = new UDPQuery(ip, port);
    query->moveToThread(m_receiver_thread);

    /// take care that objects are deleted
    connect(m_receiver_thread, SIGNAL(started()), query, SLOT(start()));
    connect(m_receiver_thread, SIGNAL(finished()), query, SLOT(deleteLater()));
    connect(m_receiver_thread, SIGNAL(finished()), m_receiver_thread, SLOT(deleteLater()));
    connect(query, SIGNAL(received_visualization(QString,QString)), this, SLOT(add_vis_name_uri_thumb(QString,QString)));
    connect(query, SIGNAL(refreshing_list()), this, SLOT(on_refreshing_list()));

    m_receiver_thread->start();
}

void SimulationGallery::on_refreshing_list()
{
    this->clear();
    this->m_vis_index_uri.clear();
    m_num_slides = 0;
}

void SimulationGallery::add_vis_name_uri_thumb(QString name, QString uri/*, QImage thumb*/)
{
    qDebug() << "Called add vis name uri thumb.";
    qDebug() << "Adding Name/URI " << name << " / " << uri;

    QImage image(400, 200, QImage::Format_RGB32);
    QPainter painter(&image);

    for (int x = 0; x < 400; x++)
    {
        for (int y = 0; y < 200; y++)
        {
            if ((x < 10 || x > 389) || (y < 4 || y > 195))
                image.setPixel(x, y, std::numeric_limits<uint>::max() - 1);
            else
                image.setPixel(x, y, 1);
        }
    }

    QFont font("Sans Serif", 17);
    QPen pen(QColor::fromRgb(255,255,255));

    painter.setFont(font);
    painter.setPen(pen);

    painter.drawText(30, 120, 380, 30, 0, name);
    painter.drawText(30, 160, 380, 30, 0, uri);

    this->addSlide(image);

    m_vis_index_uri[m_num_slides] = uri;
    m_num_slides++;
}

void SimulationGallery::keyPressEvent(QKeyEvent * event)
{
    if (event->key() == Qt::Key_F12 || event->key() == Qt::Key_Escape)
    {
        event->accept();
        emit on_hide_me();
    }

    /// connect to the choosen visualization
    if (event->key() == Qt::Key_Enter || event->key() == Qt::Key_Return)
    {
        QString uri = m_vis_index_uri[this->centerIndex()];

        qDebug() << "Enter pressed, connecting to " << uri;

        emit connect_to(uri);
        emit on_hide_me();
    }

    //if (event->key() == Qt::Key_A)
    //    this->add_vis_name_uri_thumb("PIConGPU 2", "riv://149.220.4.50:52000/PIConGPU");

    PictureFlow::keyPressEvent(event);
}
