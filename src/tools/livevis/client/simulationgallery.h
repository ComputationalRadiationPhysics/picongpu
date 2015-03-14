#ifndef SIMULATIONGALLERY_H
#define SIMULATIONGALLERY_H

#include <QWidget>
#include <QKeyEvent>
#include <QThread>
#include "pictureflow.h"


namespace Ui {
class SimulationGallery;
}

class SimulationGallery : public PictureFlow
{
    Q_OBJECT

public:

    explicit SimulationGallery(QWidget * parent = 0);
    ~SimulationGallery();

    void startQuery(std::string ip, int port);

public slots:

    void add_vis_name_uri_thumb(QString name, QString uri/*, QImage thumb*/);
    void on_refreshing_list();

signals:

    void on_hide_me();
    void connect_to(QString uri);

protected:

    void keyPressEvent(QKeyEvent * event);

private:

    Ui::SimulationGallery * ui;

    QThread * m_receiver_thread;

    /** save a list of available visualizations */
    std::map<int, QString> m_vis_index_uri;
    int m_num_slides;

    /** Address and info portnumber of visualization server */
    std::string m_visserver_ip;
    int m_visserver_infoport;
};

#endif // SIMULATIONGALLERY_H
