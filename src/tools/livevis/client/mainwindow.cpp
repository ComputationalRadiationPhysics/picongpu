/**
 * Copyright 2013-2016 Benjamin Schneider, Rene Widera, Axel Huebl
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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <unistd.h>

#include <the/exception.h>
#include <the/text/string_builder.h>

#include "message_ids.hpp"

// helper
QString formatNumberReadable( double x )
{
    if( x > 1.e12 )
      return QString::number(x/1e12,'f',0) + " Trillion";
    if( x > 1.e9 )
      return QString::number(x/1e9,'f',0)  + " Billion";
    if( x > 1.e6 )
      return QString::number(x/1e6,'f',0)  + " Million";
    if( x > 1.e3 )
      return QString::number(x/1e3,'f',0)  + " Thousand";

    return QString::number(x,'f',0);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    rivlib::image_stream_connection::listener(),
    rivlib::control_connection::listener(),
    m_imgStream(nullptr),
    m_controlConn(nullptr),
    m_currentTFA(nullptr),
    m_currentTFB(nullptr),
    m_weighting(1.f),
    m_isSimRunning(false),
    m_write_png_images(false)
{
    ui->setupUi(this);

    m_uiVisible = true;

    connect(ui->centralWidget, SIGNAL(trigger_ui()), this, SLOT(triggerUi()));

    m_dataSidebarOnScreen = true;
    m_transferfuncSidebarOnScreen = true;
    m_settingsSidebarOnScreen = true;
    m_infoBarOnScreen = true;
    m_clipBarOnScreen = true;

    LEFT_X_SHOWN = 0;
    LEFT_X_HIDDEN = -ui->dataSidebar->width() + 40;

    RIGHT_X_SHOWN = this->width() - ui->clipSidebar->width();
    RIGHT_X_HIDDEN = this->width() - 40;

    m_aniDataSidebar = new QPropertyAnimation(ui->dataSidebar, "pos");
    m_aniTransferfuncSidebar = new QPropertyAnimation(ui->transferfuncSidebar, "pos");
    m_aniSettingsSidebar = new QPropertyAnimation(ui->settingsSidebar, "pos");
    m_aniInfoBar = new QPropertyAnimation(ui->infoBar, "pos");
    m_aniClipBar = new QPropertyAnimation(ui->clipSidebar, "pos");

    /// connect trigger signals from GLWidget with animation
    connect(ui->centralWidget, SIGNAL(trigger_dataSidebar()), this, SLOT(triggerDataSidebar()));
    connect(ui->centralWidget, SIGNAL(trigger_transferfuncSidebar()), this, SLOT(triggerTransferfuncSidebar()));
    connect(ui->centralWidget, SIGNAL(trigger_settingsSidebar()), this, SLOT(triggerSettingsSidebar()));
    connect(ui->centralWidget, SIGNAL(trigger_infoBar()), this, SLOT(triggerInfoBar()));
    connect(ui->centralWidget, SIGNAL(trigger_clipBar()), this, SLOT(triggerClipBar()));

    /// init RIVLib connection and stream
    this->m_controlConn = rivlib::control_connection::create();
    this->m_controlConn->add_listener(this);

    this->m_imgStream = rivlib::image_stream_connection::create();
    this->m_imgStream->add_listener(this);

    /// init simulation gallery
    m_simulationGallery = new SimulationGallery(this);
    m_simulationGallery->setGeometry(0, 0, this->width(), this->height());
    m_simulationGalleryOnScreen = true;
    m_simulationGallery->show();
    m_simulationGallery->setReflectionEffect(PictureFlow::BlurredReflection);
    m_simulationGallery->setFocus();

    connect(ui->centralWidget, SIGNAL(show_simulationGallery()), this, SLOT(triggerSimulationGallery()));
    connect(m_simulationGallery, SIGNAL(on_hide_me()), this, SLOT(triggerSimulationGallery()));

    /// enable changing background color
    connect(ui->lblBGColor, SIGNAL(clicked()), this, SLOT(changeBackgroundcolor()));

    /// enable connecting to a visualization via its RIV URI
    connect(m_simulationGallery, SIGNAL(connect_to(QString)), this, SLOT(connectToURI(QString)));

    /// enable steering of simulation by interaction with mouse and keyboard
    connect(ui->centralWidget, SIGNAL(send_message(uint,uint,const void*)), this, SLOT(sendMessage(uint,uint,const void*)));

    /// add received data sources to the list of available ones
    connect(this, SIGNAL(on_received_datasource(QString)), this, SLOT(received_datasource(QString)));

    /// reset to initial values
    this->reset_connection();

    /// initialize Transferfunctions
    m_currentTFA = new RedGreenTransferFunction();
    m_currentTFB = new TemperatureTransferFunction();

    connect(this, SIGNAL(on_transferFunctionA_changed()), this, SLOT(transferFunctionA_changed()));
    connect(this, SIGNAL(on_transferFunctionB_changed()), this, SLOT(transferFunctionB_changed()));

    emit on_transferFunctionA_changed();
    emit on_transferFunctionB_changed();

    /// enable display of current timestep
    connect(this, SIGNAL(on_received_timestep(int)), this, SLOT(received_timestep(int)));

    /// fps display
    connect(this, SIGNAL(on_received_fps(float)), this, SLOT(received_fps(float)));
    connect(this, SIGNAL(on_received_rfps(float)), this, SLOT(received_rfps(float)));

    /// num GPU/cells/particles display
    connect(this, SIGNAL(on_received_numGPUs(int)), this, SLOT(received_numGPUs(int)));
    connect(this, SIGNAL(on_received_numCells(double)), this, SLOT(received_numCells(double)));
    connect(this, SIGNAL(on_received_numParticles(double)), this, SLOT(received_numParticles(double)));

    /// enable play/pause of simulation
    connect(ui->centralWidget, SIGNAL(play_pause_sim()), this, SLOT(playPauseSim()));
    connect(ui->btnPlayPause, SIGNAL(clicked()), this, SLOT(playPauseSim()));

    /// enable volume clipping
    m_clip[0] = 0.0f;   // min x
    m_clip[1] = 0.0f;   // min y
    m_clip[2] = 0.0f;   // min z
    m_clip[3] = 1.0f;   // max x
    m_clip[4] = 1.0f;   // max y
    m_clip[5] = 1.0f;   // max z

    connect(ui->sldXMin, SIGNAL(valueChanged(int)), this, SLOT(clippingChanged()));
    connect(ui->sldYMin, SIGNAL(valueChanged(int)), this, SLOT(clippingChanged()));
    connect(ui->sldZMin, SIGNAL(valueChanged(int)), this, SLOT(clippingChanged()));
    connect(ui->sldXMax, SIGNAL(valueChanged(int)), this, SLOT(clippingChanged()));
    connect(ui->sldYMax, SIGNAL(valueChanged(int)), this, SLOT(clippingChanged()));
    connect(ui->sldZMax, SIGNAL(valueChanged(int)), this, SLOT(clippingChanged()));

    connect(m_aniClipBar, SIGNAL(finished()), this, SLOT(refresh_glWidget()));
    connect(m_aniDataSidebar, SIGNAL(finished()), this, SLOT(refresh_glWidget()));
    connect(m_aniInfoBar, SIGNAL(finished()), this, SLOT(refresh_glWidget()));
    connect(m_aniSettingsSidebar, SIGNAL(finished()), this, SLOT(refresh_glWidget()));
    connect(m_aniTransferfuncSidebar, SIGNAL(finished()), this, SLOT(refresh_glWidget()));
}

void MainWindow::refresh_glWidget()
 {
    QApplication::sendEvent(ui->centralWidget, new QWheelEvent(QPoint(0,0), 0, Qt::MouseButton::NoButton, Qt::KeyboardModifier::NoModifier));
 }

MainWindow::~MainWindow()
{
    delete ui;

    this->m_imgStream->remove_listener(this);
    if (this->m_imgStream->get_status() != rivlib::control_connection::status::not_connected)
    {
        this->m_imgStream->disconnect();
    }
    this->m_imgStream.reset();

    this->m_controlConn->remove_listener(this);
    if (this->m_controlConn->get_status() != rivlib::control_connection::status::not_connected)
    {
        this->m_controlConn->disconnect();
    }
    this->m_controlConn.reset();

    if (m_aniDataSidebar) delete m_aniDataSidebar;
    if (m_aniInfoBar) delete m_aniInfoBar;
    if (m_aniSettingsSidebar) delete m_aniSettingsSidebar;
    if (m_aniTransferfuncSidebar) delete m_aniTransferfuncSidebar;
    if (m_aniClipBar) delete m_aniClipBar;
}

void MainWindow::on_image_data(rivlib::image_stream_connection::ptr comm, uint32_t width, uint32_t height, const void * rgbpix) throw()
{
#if (MEASURE_TIME_CPUCLOCK == 1)
    clock_gettime(CLOCK_REALTIME, &m_stop);
    std::cout << "CPU PROFILER (Latency): " << (m_stop.tv_sec - m_start.tv_sec) << " s and " << (m_stop.tv_nsec - m_start.tv_nsec) << " ns." << std::endl;
#endif

    ui->centralWidget->update_image(width, height, rgbpix);
    ui->centralWidget->update();
}

void MainWindow::on_error(rivlib::image_stream_connection::ptr comm, const char * msg) throw()
{
    std::cout << "Image Stream Error: " << msg << std::endl;
}

void MainWindow::on_connected(rivlib::image_stream_connection::ptr comm) throw()
{
    std::cout << "Image Stream connected!" << std::endl;
}

void MainWindow::on_disconnected(rivlib::image_stream_connection::ptr comm) throw()
{
    std::cout << "Image Stream disconnected!" << std::endl;
}

void MainWindow::on_msg(rivlib::control_connection::ptr comm, unsigned int id, unsigned int size, const void * data) throw()
{
    try {
        switch (id)
        {
        case static_cast<unsigned int>(rivlib::message_id::data_channels):
        {
            size_t pos = 0;
            if (size < pos + sizeof(uint32_t)) throw the::exception(__FILE__, __LINE__);
            uint32_t cnt = *the::as_at<uint32_t>(data, pos);
            pos += sizeof(uint32_t);

            printf("Provider publishes %u data channels:\n", static_cast<unsigned int>(cnt));

            std::string sel_chan_name;
            uint8_t sel_chan_qual = 0;
            uint16_t sel_chan_type = 0;
            uint16_t sel_chan_subtype = 0;

            for (uint32_t i = 0; i < cnt; ++i)
            {
                if (size < pos + sizeof(uint16_t)) throw the::exception(__FILE__, __LINE__);
                uint16_t namelen = *the::as_at<uint16_t>(data, pos);
                pos += sizeof(uint16_t);
                if (size < pos + namelen + sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint8_t)) throw the::exception(__FILE__, __LINE__);
                std::string name(the::as_at<char>(data, pos), namelen);
                pos += namelen;
                uint16_t chan_type = *the::as_at<uint16_t>(data, pos);
                pos += sizeof(uint16_t);
                uint16_t chan_subtype = *the::as_at<uint16_t>(data, pos);
                pos += sizeof(uint16_t);
                uint8_t chan_quality = *the::as_at<uint8_t>(data, pos);
                pos += sizeof(uint8_t);

                printf("\t%s (%u,%u;%u)\n",
                    name.c_str(),
                    static_cast<unsigned int>(chan_type),
                    static_cast<unsigned int>(chan_subtype),
                    static_cast<unsigned int>(chan_quality));

                // select best image_stream channel
                if (static_cast<rivlib::data_channel_type>(chan_type) == rivlib::data_channel_type::image_stream) {
                    rivlib::data_channel_image_stream_subtype subtype = static_cast<rivlib::data_channel_image_stream_subtype>(chan_subtype);

                    // skip unsupported streams
                    if (!this->m_imgStream->is_supported(subtype)) continue;

                    if (subtype == rivlib::data_channel_image_stream_subtype::rgb_mjpeg) {
                        // subtype is supported by this client o_O
                        if (sel_chan_qual < chan_quality) {
                            sel_chan_name = name;
                            sel_chan_type = chan_type;
                            sel_chan_subtype = chan_subtype;
                            sel_chan_qual = chan_quality;
                        }
                    }
                }
            }

            if (sel_chan_name.empty()) {
                printf("No suitable image_stream data channel available\n");

            } else {
                printf("Selected \"%s\" image_stream data channel of type %u\n", sel_chan_name.c_str(), sel_chan_subtype);
                this->m_imgStream->disconnect(true);

                size_t uri_len = this->m_controlConn->make_data_channel_uri(sel_chan_name.c_str(), sel_chan_type, sel_chan_subtype, nullptr, 0);
                char *buf = new char[uri_len];
                uri_len = this->m_controlConn->make_data_channel_uri(sel_chan_name.c_str(), sel_chan_type, sel_chan_subtype, buf, uri_len);
                std::string uri(buf, uri_len);
                delete[] buf;

                this->m_imgStream->connect(uri.c_str());
            }

        } break;

        case RIVLIB_USERMSG + TimeStep: {
            unsigned int timestep = reinterpret_cast<const uint32_t*>(data)[0];
            emit on_received_timestep(timestep);
        } break;
        case RIVLIB_USERMSG + FPS: {
            float fps = reinterpret_cast<const  float*>(data)[0];
            emit on_received_fps(fps);
        } break;
        case RIVLIB_USERMSG + RenderFPS: {
            float fps = reinterpret_cast<const  float*>(data)[0];
            emit on_received_rfps(fps);
        } break;

        case RIVLIB_USERMSG + NumGPUs: {
            int numGPUs = reinterpret_cast<const int64_t*>(data)[0];
            emit on_received_numGPUs(numGPUs);
        } break;
        case RIVLIB_USERMSG + NumCells: {
            double numCells = reinterpret_cast<const int64_t*>(data)[0];
            emit on_received_numCells(numCells);
        } break;
        case RIVLIB_USERMSG + NumParticles: {
            double numParticles = reinterpret_cast<const int64_t*>(data)[0];
            emit on_received_numParticles(numParticles);
        } break;

        case RIVLIB_USERMSG + AvailableDataSource: {
            char * c = new char[size + 1];
            memcpy(c, data, size);
            c[size] = 0;

            emit on_received_datasource(QString(c));

            delete [] c;
        } break;

        case RIVLIB_USERMSG + VisibleSimulationArea: {
            ui->centralWidget->set_simulation_area(reinterpret_cast<const float*>(data)[0],
                                                   reinterpret_cast<const float*>(data)[1],
                                                   reinterpret_cast<const float*>(data)[2],
                                                   reinterpret_cast<const float*>(data)[3],
                                                   reinterpret_cast<const float*>(data)[4],
                                                   reinterpret_cast<const float*>(data)[5]);
        } break;

        default:
            printf("Ctrl Msg received: %u (%u bytes)\n", id, static_cast<unsigned int>(size));
            break;
        }

    } catch(const the::exception& ex) {
        fprintf(stderr, "Ctrl-Message error: %s (%s, %d)\n", ex.get_msg_astr(), ex.get_file(), ex.get_line());
        //this->on_disconnect_clicked();
    } catch(...) {
        fprintf(stderr, "Ctrl-Message error: unexpected exception\n");
        //this->on_disconnect_clicked();
    }
}

void MainWindow::on_error(rivlib::control_connection::ptr comm, const char * msg) throw()
{
    std::cout << "Control Connection: " << msg << std::endl;
}

void MainWindow::on_connected(rivlib::control_connection::ptr comm) throw()
{
    std::cout << "Control Connection connected!" << std::endl;

    this->sendMessage(RequestDataSources, 0, nullptr);
    this->on_pushButton_clicked();

    this->m_controlConn->send(static_cast<unsigned int>(rivlib::message_id::query_data_channels), 0, nullptr);
}

void MainWindow::on_disconnected(rivlib::control_connection::ptr comm) throw()
{
    std::cout << "Control Connection disconnected!" << std::endl;

    try {
        this->m_imgStream->disconnect(true);
    } catch(...) {
    }
}

void MainWindow::resizeEvent(QResizeEvent * re)
{
    this->layout()->setEnabled(true);
    ui->centralWidget->layout()->setEnabled(true);
    ui->dockWidget->layout()->setEnabled(true);

    m_simulationGallery->setGeometry(0, 0, this->width(), this->height());

    RIGHT_X_SHOWN = this->width() - ui->clipSidebar->width();
    RIGHT_X_HIDDEN = this->width() - 40;

    this->correct_sidebar_positions();

    if (m_simulationGalleryOnScreen) m_simulationGallery->setFocus();
}

void MainWindow::reset_connection()
{
    /// clean up potential previous connections
    this->m_controlConn->disconnect(false);

    /// clean up data source sidebar
    ui->cmbDatasourceA->clear();
    ui->cmbDatasourceB->clear();

    ui->cmbDatasourceA->addItem("None");
    ui->cmbDatasourceB->addItem("None");

    /// reset transferfunction sidebar
    ui->rdo2HueA->setChecked(false);
    ui->rdoRedGreenA->setChecked(true);
    ui->rdoTempA->setChecked(false);

    ui->rdo2HueB->setChecked(false);
    ui->rdoRedGreenB->setChecked(false);
    ui->rdoTempB->setChecked(true);

    /// reset Rendering/Compositing Settings
    ui->rdoAlphaBlending->setChecked(true);
    ui->rdoMIP->setChecked(false);
    ui->rdoIsoSurface->setChecked(false);

    m_backgroundColor[0] = m_backgroundColor[1] = m_backgroundColor[2] = 1.0f;

    QImage image(1, 1, QImage::Format_RGB32);
    image.setPixel(0, 0, std::numeric_limits<uint>::max());

    ui->lblBGColor->setPixmap(QPixmap::fromImage(image));
}

void MainWindow::sendMessage(unsigned int id, unsigned int size, const void * data)
{
#if (MEASURE_TIME_CPUCLOCK == 1)
    clock_gettime(CLOCK_REALTIME, &m_start);
#endif

    /// send message to server via control connection
    this->m_controlConn->send(RIVLIB_USERMSG + id, size, data);

    //std::cout << "Sending Message via RIV " << id << " " << size << std::endl;
}

void MainWindow::connectToURI(QString uri)
{
    std::cout << "Connecting..." << std::endl;

    try {
        reset_connection();
        this->m_controlConn->connect(uri.toStdString().c_str());
        this->m_imgStream->disconnect(false);
    } catch(const the::exception& ex) {
        fprintf(stderr, "Connect-Message error: %s (%s, %d)\n", ex.get_msg_astr(), ex.get_file(), ex.get_line());
        //this->m_imgStream->disconnect(true);
    } catch(...) {
        fprintf(stderr, "Connect-Message error: unexpected exception\n");
        //this->m_imgStream->disconnect(true);
    }
}

void MainWindow::received_datasource(QString source)
{
    ui->cmbDatasourceA->addItem(source);
    ui->cmbDatasourceB->addItem(source);

    //std::cout << "Datasource " << source.toStdString() << " added." << std::endl;
}

void MainWindow::received_timestep(int step)
{
    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);
    ui->dockWidget->layout()->setEnabled(false);

    ui->lblTimestep->setText(QString::number(step));
}

void MainWindow::received_fps(float fps)
{
    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);
    ui->dockWidget->layout()->setEnabled(false);

    ui->lblFPS->setText(QString::number(fps,'f',1));
}

void MainWindow::received_rfps(float fps)
{
    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);
    ui->dockWidget->layout()->setEnabled(false);

    ui->lblRFPS->setText(QString::number(fps,'f',1));
}

void MainWindow::received_numGPUs(int numGPUs)
{
    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);
    ui->dockWidget->layout()->setEnabled(false);

    ui->lblStatusGPUs->setText(QString("%1 GPUs").arg(numGPUs));
}

void MainWindow::received_numCells(double numCells)
{
    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);
    ui->dockWidget->layout()->setEnabled(false);

    ui->lblStatusCells->setText(formatNumberReadable(numCells) + " Cells");
}

void MainWindow::received_numParticles(double numParticles)
{
    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);
    ui->dockWidget->layout()->setEnabled(false);

    ui->lblStatusParticles->setText(formatNumberReadable(numParticles) + " Particles");
}

void MainWindow::changeBackgroundcolor()
{
    QColor color = QColorDialog::getColor( QColor(m_backgroundColor[0] * 255.0f,
                                                  m_backgroundColor[1] * 255.0f,
                                                  m_backgroundColor[2] * 255.0f),
                                           this, QString("Pick a Background Color") );

    m_backgroundColor[0] = static_cast<float>(color.red()) / 255.0f;
    m_backgroundColor[1] = static_cast<float>(color.green()) / 255.0f;
    m_backgroundColor[2] = static_cast<float>(color.blue()) / 255.0f;

    unsigned int colorUint = color.red() << 16 | color.green() << 8 | color.blue();

    QImage image(1, 1, QImage::Format_RGB32);
    image.setPixel(0, 0, colorUint);

    this->layout()->setEnabled(false);
    ui->centralWidget->layout()->setEnabled(false);

    ui->lblBGColor->setPixmap(QPixmap::fromImage(image));
    ui->centralWidget->set_background_color(m_backgroundColor[0], m_backgroundColor[1], m_backgroundColor[2]);

    QWidget* titleWidget = new QWidget(this); /* where this a QMainWindow object */
    ui->dockWidget->setTitleBarWidget( titleWidget );
    ui->dockWidgetContents->setStyleSheet("background-color: rgb(" + QString::number(m_backgroundColor[0]*255) +
                                                               "," + QString::number(m_backgroundColor[1]*255) +
                                                               "," + QString::number(m_backgroundColor[2]*255) +
                                                               "," + ");");


    // status bar labels
    QPalette lblTxtPalGPUs = QPalette(ui->lblStatusGPUs->palette());
    lblTxtPalGPUs.setColor(ui->lblStatusGPUs->foregroundRole(), QColor(255-m_backgroundColor[0]* 255.0f, 255-m_backgroundColor[1]* 255.0f, 255-m_backgroundColor[2]* 255.0f));
    ui->lblStatusGPUs->setPalette(lblTxtPalGPUs);

    QPalette lblTxtPalCells = QPalette(ui->lblStatusCells->palette());
    lblTxtPalCells.setColor(ui->lblStatusCells->foregroundRole(), QColor(255-m_backgroundColor[0]* 255.0f, 255-m_backgroundColor[1]* 255.0f, 255-m_backgroundColor[2]* 255.0f));
    ui->lblStatusCells->setPalette(lblTxtPalCells);

    QPalette lblTxtPalParticles = QPalette(ui->lblStatusParticles->palette());
    lblTxtPalParticles.setColor(ui->lblStatusParticles->foregroundRole(), QColor(255-m_backgroundColor[0]* 255.0f, 255-m_backgroundColor[1]* 255.0f, 255-m_backgroundColor[2]* 255.0f));
    ui->lblStatusParticles->setPalette(lblTxtPalParticles);

    //emit backgroundColor_updated(colorUint);
    this->sendMessage(BackgroundColor, 3 * sizeof(float), m_backgroundColor);
}

void MainWindow::triggerSimulationGallery()
{
    if (m_simulationGalleryOnScreen)
    {
        m_simulationGallery->hide();
        m_simulationGalleryOnScreen = false;
        ui->centralWidget->setFocus();
    }
    else
    {
        m_simulationGallery->show();
        m_simulationGalleryOnScreen = true;
        m_simulationGallery->setFocus();
    }
}

void MainWindow::hideAllSidebars()
{
    m_dataSidebarOnScreen = m_transferfuncSidebarOnScreen = m_settingsSidebarOnScreen = m_infoBarOnScreen = true;

    triggerDataSidebar();
    triggerTransferfuncSidebar();
    triggerSettingsSidebar();
    triggerInfoBar();
    triggerClipBar();
}

void MainWindow::triggerDataSidebar()
{
    /*if (m_dataSidebarOnScreen)
    {
        ui->dataSidebar->hide();
        m_dataSidebarOnScreen = false;
    }
    else
    {
        ui->dataSidebar->show();
        m_dataSidebarOnScreen = true;
    }*/
    m_aniDataSidebar->setDuration(500);

    if (m_dataSidebarOnScreen)
    {
        m_aniDataSidebar->setEndValue(QPoint(LEFT_X_HIDDEN, ui->dataSidebar->pos().y()));
        m_aniDataSidebar->setEasingCurve(QEasingCurve::InSine);
        m_dataSidebarOnScreen = false;
    }
    else
    {
        m_aniDataSidebar->setEndValue(QPoint(LEFT_X_SHOWN, ui->dataSidebar->pos().y()));
        m_aniDataSidebar->setEasingCurve(QEasingCurve::OutSine);
        m_dataSidebarOnScreen = true;
    }

    m_aniDataSidebar->start();
}

void MainWindow::triggerTransferfuncSidebar()
{
    m_aniTransferfuncSidebar->setDuration(500);

    if (m_transferfuncSidebarOnScreen)
    {
        m_aniTransferfuncSidebar->setEndValue(QPoint(LEFT_X_HIDDEN, ui->transferfuncSidebar->pos().y()));
        m_aniTransferfuncSidebar->setEasingCurve(QEasingCurve::InSine);
        m_transferfuncSidebarOnScreen = false;
    }
    else
    {
        m_aniTransferfuncSidebar->setEndValue(QPoint(LEFT_X_SHOWN, ui->transferfuncSidebar->pos().y()));
        m_aniTransferfuncSidebar->setEasingCurve(QEasingCurve::OutSine);
        m_transferfuncSidebarOnScreen = true;
    }

    m_aniTransferfuncSidebar->start();
}

void MainWindow::triggerSettingsSidebar()
{
    m_aniSettingsSidebar->setDuration(500);

    if (m_settingsSidebarOnScreen)
    {
        m_aniSettingsSidebar->setEndValue(QPoint(LEFT_X_HIDDEN, ui->settingsSidebar->pos().y()));
        m_aniSettingsSidebar->setEasingCurve(QEasingCurve::InSine);
        m_settingsSidebarOnScreen = false;
    }
    else
    {
        m_aniSettingsSidebar->setEndValue(QPoint(LEFT_X_SHOWN, ui->settingsSidebar->pos().y()));
        m_aniSettingsSidebar->setEasingCurve(QEasingCurve::OutSine);
        m_settingsSidebarOnScreen = true;
    }

    m_aniSettingsSidebar->start();
}

void MainWindow::triggerInfoBar()
{
    m_aniInfoBar->setDuration(500);

    if (m_infoBarOnScreen)
    {
        m_aniInfoBar->setEndValue(QPoint(RIGHT_X_HIDDEN, ui->infoBar->pos().y()));
        m_aniInfoBar->setEasingCurve(QEasingCurve::InSine);
        m_infoBarOnScreen = false;
    }
    else
    {
        m_aniInfoBar->setEndValue(QPoint(RIGHT_X_SHOWN, ui->infoBar->pos().y()));
        m_aniInfoBar->setEasingCurve(QEasingCurve::OutSine);
        m_infoBarOnScreen = true;
    }

    m_aniInfoBar->start();
}

void MainWindow::triggerClipBar()
{
    m_aniClipBar->setDuration(500);

    if (m_clipBarOnScreen)
    {
        m_aniClipBar->setEndValue(QPoint(RIGHT_X_HIDDEN, ui->clipSidebar->pos().y()));
        m_aniClipBar->setEasingCurve(QEasingCurve::InSine);
        m_clipBarOnScreen = false;
    }
    else
    {
        m_aniClipBar->setEndValue(QPoint(RIGHT_X_SHOWN, ui->clipSidebar->pos().y()));
        m_aniClipBar->setEasingCurve(QEasingCurve::OutSine);
        m_clipBarOnScreen = true;
    }

    m_aniClipBar->start();
}

void MainWindow::triggerUi()
{
    if (m_uiVisible)
    {
        ui->dataSidebar->hide();
        ui->transferfuncSidebar->hide();
        ui->settingsSidebar->hide();
        //ui->infoBar->hide();
        ui->clipSidebar->hide();

        m_uiVisible = false;
    }
    else
    {
        ui->dataSidebar->show();
        ui->transferfuncSidebar->show();
        ui->settingsSidebar->show();
        //ui->infoBar->show();
        ui->clipSidebar->show();

        this->correct_sidebar_positions();

        m_uiVisible = true;
    }

    this->refresh_glWidget();
}

void MainWindow::playPauseSim()
{
    if (m_isSimRunning)
    {
        m_isSimRunning = false;
        this->sendMessage(SimPause, 0, nullptr);
        ui->btnPlayPause->setIcon(QIcon(":/icons/play_button.png"));
    }
    else
    {
        m_isSimRunning = true;
        this->sendMessage(SimPlay, 0, nullptr);
        ui->btnPlayPause->setIcon(QIcon(":/icons/pause_button.png"));
    }
}

void MainWindow::clippingChanged()
{
    m_clip[1] = static_cast<float>(ui->sldXMin->value()) * 0.01f;
    m_clip[0] = static_cast<float>(ui->sldYMin->value()) * 0.01f;
    m_clip[2] = static_cast<float>(ui->sldZMin->value()) * 0.01f;

    m_clip[4] = static_cast<float>(ui->sldXMax->value()) * 0.01f;
    m_clip[3] = static_cast<float>(ui->sldYMax->value()) * 0.01f;
    m_clip[5] = static_cast<float>(ui->sldZMax->value()) * 0.01f;

    this->sendMessage(Clipping, 6 * sizeof(float), m_clip);
}

void MainWindow::on_cmbDatasourceA_currentIndexChanged(const QString &arg1)
{
    ui->grpColorscaleA->setTitle(arg1);
    this->sendMessage(DataSourceA, arg1.toStdString().size(), arg1.toStdString().c_str());
}

void MainWindow::on_cmbDatasourceB_currentIndexChanged(const QString &arg1)
{
    ui->grpColorscaleB->setTitle(arg1);
    this->sendMessage(DataSourceB, arg1.toStdString().size(), arg1.toStdString().c_str());
}

void MainWindow::on_rdoRedGreenA_toggled(bool checked)
{
    if (checked)
    {
        if (m_currentTFA) delete m_currentTFA;
        m_currentTFA = new RedGreenTransferFunction();

        /// update the slider to reflect the current state of the chosen TF
        ui->sldOffsetXTFA->setValue(m_currentTFA->getOffsetX() * 100.0f);
        ui->sldOffsetYTFA->setValue(m_currentTFA->getOffsetY() * 100.0f);
        ui->sldSlopeTFA->setValue(m_currentTFA->getSlope() * 10.0f);

        emit on_transferFunctionA_changed();
    }
}

void MainWindow::on_rdoTempA_toggled(bool checked)
{
    if (checked)
    {
        if (m_currentTFA) delete m_currentTFA;
        m_currentTFA = new TemperatureTransferFunction();

        /// update the slider to reflect the current state of the chosen TF
        ui->sldOffsetXTFA->setValue(m_currentTFA->getOffsetX() * 100.0f);
        ui->sldOffsetYTFA->setValue(m_currentTFA->getOffsetY() * 100.0f);
        ui->sldSlopeTFA->setValue(m_currentTFA->getSlope() * 10.0f);

        emit on_transferFunctionA_changed();
    }
}

void MainWindow::on_rdo2HueA_toggled(bool checked)
{
    if (checked)
    {
        if (m_currentTFA) delete m_currentTFA;
        m_currentTFA = new TwoHueTransferFunction();

        /// update the slider to reflect the current state of the chosen TF
        ui->sldOffsetXTFA->setValue(m_currentTFA->getOffsetX() * 100.0f);
        ui->sldOffsetYTFA->setValue(m_currentTFA->getOffsetY() * 100.0f);
        ui->sldSlopeTFA->setValue(m_currentTFA->getSlope() * 10.0f);

        emit on_transferFunctionA_changed();
    }
}

void MainWindow::on_rdoRedGreenB_toggled(bool checked)
{
    if (checked)
    {
        if (m_currentTFB) delete m_currentTFB;
        m_currentTFB = new RedGreenTransferFunction();

        /// update the slider to reflect the current state of the chosen TF
        ui->sldOffsetXTFB->setValue(m_currentTFB->getOffsetX() * 100.0f);
        ui->sldOffsetYTFB->setValue(m_currentTFB->getOffsetY() * 100.0f);
        ui->sldSlopeTFB->setValue(m_currentTFB->getSlope() * 10.0f);

        emit on_transferFunctionB_changed();
    }
}

void MainWindow::on_rdoTempB_toggled(bool checked)
{
    if (checked)
    {
        if (m_currentTFB) delete m_currentTFB;
        m_currentTFB = new TemperatureTransferFunction();

        /// update the slider to reflect the current state of the chosen TF
        ui->sldOffsetXTFB->setValue(m_currentTFB->getOffsetX() * 100.0f);
        ui->sldOffsetYTFB->setValue(m_currentTFB->getOffsetY() * 100.0f);
        ui->sldSlopeTFB->setValue(m_currentTFB->getSlope() * 10.0f);

        emit on_transferFunctionB_changed();
    }
}

void MainWindow::on_rdo2HueB_toggled(bool checked)
{
    if (checked)
    {
        if (m_currentTFB) delete m_currentTFB;
        m_currentTFB = new TwoHueTransferFunction();

        /// update the slider to reflect the current state of the chosen TF
        ui->sldOffsetXTFB->setValue(m_currentTFB->getOffsetX() * 100.0f);
        ui->sldOffsetYTFB->setValue(m_currentTFB->getOffsetY() * 100.0f);
        ui->sldSlopeTFB->setValue(m_currentTFB->getSlope() * 10.0f);

        emit on_transferFunctionB_changed();
    }
}

void MainWindow::transferFunctionA_changed()
{
    /// sample the TF
    float4 * sampled = new float4[TF_RESOLUTION];

    for (int i = 0; i < TF_RESOLUTION; ++i)
    {
        sampled[i] = m_currentTFA->sample( static_cast<float>(i) / static_cast<float>(TF_RESOLUTION - 1) );
    }

    /// update color scale label
    QImage image(1, TF_RESOLUTION, QImage::Format_RGB32);

    for (int y = 0; y < TF_RESOLUTION; ++y)
    {
        QColor color(sampled[y].r * 255.0f, sampled[y].g * 255.0f, sampled[y].b * 255.0f);
        unsigned int colorUint = color.red() << 16 | color.green() << 8 | color.blue();
        image.setPixel(0, TF_RESOLUTION - y - 1, colorUint);
    }

    QPixmap pm(QPixmap::fromImage(image));

    static bool first = true;
    if (first)
        first = false;
    else
    {
        this->layout()->setEnabled(false);
        ui->centralWidget->layout()->setEnabled(false);
    }

    ui->lblTFA->setPixmap(pm);

    /// send new TF via RIV
    this->sendMessage(TransferFunctionA, sizeof(float4) * TF_RESOLUTION, sampled);

    delete [] sampled;

    /*std::cout << "DSBar X: " << ui->dataSidebar->pos().x() << std::endl;
    std::cout << "TFBar X: " << ui->transferfuncSidebar->pos().x() << std::endl;
    std::cout << "RSBar X: " << ui->settingsSidebar->pos().x() << std::endl;

    std::cout << "TF changed Thread: " << QThread::currentThreadId() << std::endl;*/

    emit on_correct_sidebar_positions();
}

void MainWindow::transferFunctionB_changed()
{
    /// sample the TF
    float4 * sampled = new float4[TF_RESOLUTION];

    for (int i = 0; i < TF_RESOLUTION; ++i)
    {
        sampled[i] = m_currentTFB->sample( static_cast<float>(i) / static_cast<float>(TF_RESOLUTION - 1) );
    }

    /// update color scale label
    QImage image(1, TF_RESOLUTION, QImage::Format_RGB32);

    for (int y = 0; y < TF_RESOLUTION; ++y)
    {
        QColor color(sampled[y].r * 255.0f, sampled[y].g * 255.0f, sampled[y].b * 255.0f);
        unsigned int colorUint = color.red() << 16 | color.green() << 8 | color.blue();
        image.setPixel(0, TF_RESOLUTION - y - 1, colorUint);
    }

    QPixmap pm(QPixmap::fromImage(image));

    static bool first = true;
    if (first)
        first = false;
    else
    {
        this->layout()->setEnabled(false);
        ui->centralWidget->layout()->setEnabled(false);
    }

    ui->lblTFB->setPixmap(pm);

    /// send new TF via RIV
    this->sendMessage(TransferFunctionB, sizeof(float4) * TF_RESOLUTION, sampled);

    delete [] sampled;
}

void MainWindow::on_sldOffsetXTFA_valueChanged(int value)
{
    m_currentTFA->setOffsetX( static_cast<float>(value) * 0.01f );

    emit on_transferFunctionA_changed();
}

void MainWindow::on_sldSlopeTFA_valueChanged(int value)
{
    m_currentTFA->setSlope( static_cast<float>(value) * 0.1f );

    emit on_transferFunctionA_changed();
}

void MainWindow::on_sldOffsetYTFA_valueChanged(int value)
{
    m_currentTFA->setOffsetY( static_cast<float>(value) * 0.005f );

    emit on_transferFunctionA_changed();
}

void MainWindow::on_sldOffsetXTFB_valueChanged(int value)
{
    m_currentTFB->setOffsetX( static_cast<float>(value) * 0.01f );

    emit on_transferFunctionB_changed();
}

void MainWindow::on_sldSlopeTFB_valueChanged(int value)
{
    m_currentTFB->setSlope( static_cast<float>(value) * 0.1f );

    emit on_transferFunctionB_changed();
}

void MainWindow::on_sldOffsetYTFB_valueChanged(int value)
{
    m_currentTFB->setOffsetY( static_cast<float>(value) * 0.005f );

    emit on_transferFunctionB_changed();
}

void MainWindow::on_sldSourceWeighting_valueChanged(int value)
{
    m_weighting = static_cast<float>(value) * 0.01f;
    this->sendMessage(Weighting, sizeof(float), &m_weighting);
}

void MainWindow::correct_sidebar_positions()
{
    //std::cout << "Correction Thread: " << QThread::currentThreadId() << std::endl;

    if (m_dataSidebarOnScreen) ui->dataSidebar->move(LEFT_X_SHOWN, ui->dataSidebar->pos().y());
    else ui->dataSidebar->move(LEFT_X_HIDDEN, ui->dataSidebar->pos().y());

    if (m_settingsSidebarOnScreen) ui->settingsSidebar->move(LEFT_X_SHOWN, ui->settingsSidebar->pos().y());
    else ui->settingsSidebar->move(LEFT_X_HIDDEN, ui->settingsSidebar->pos().y());

    if (m_transferfuncSidebarOnScreen) ui->transferfuncSidebar->move(LEFT_X_SHOWN, ui->transferfuncSidebar->pos().y());
    else ui->transferfuncSidebar->move(LEFT_X_HIDDEN, ui->transferfuncSidebar->pos().y());

    if (m_infoBarOnScreen) ui->infoBar->move(RIGHT_X_SHOWN, ui->infoBar->pos().y());
    else ui->infoBar->move(RIGHT_X_HIDDEN, ui->infoBar->pos().y());

    if (m_infoBarOnScreen) ui->clipSidebar->move(RIGHT_X_SHOWN, ui->clipSidebar->pos().y());
    else ui->clipSidebar->move(RIGHT_X_HIDDEN, ui->clipSidebar->pos().y());

    //std::cout << "DataSources is " << (m_dataSidebarOnScreen ? "ON" : "OFF") << std::endl;
}

/// DEAD CODE, darn
void MainWindow::on_sldOffsetXTFA_sliderReleased()
{ }
/// DEAD END

void MainWindow::on_pushButton_clicked()
{
    //this->sendMessage(CameraDefault, 0, nullptr);

    /// default camera position and focal point
    float3 center = ui->centralWidget->get_simulation_center();
    Camera * cam = ui->centralWidget->getCamera();

    cam->setPosition(center * 2.0f);
    cam->setFocalPoint(center);

    float pos[3], focal[3];
    pos[0] = cam->getPosition().x;
    pos[1] = cam->getPosition().y;
    pos[2] = cam->getPosition().z;
    focal[0] = cam->getFocalPoint().x;
    focal[1] = cam->getFocalPoint().y;
    focal[2] = cam->getFocalPoint().z;

    //std::cout << "CameraPosition: " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
    //std::cout << "CameraFocalPoint: " << focal[0] << " " << focal[1] << " " << focal[2] << std::endl;

        this->sendMessage(CameraDefault, 3 * sizeof(float), pos);
   // this->sendMessage(CameraPosition, 3 * sizeof(float), pos);
   // this->sendMessage(CameraFocalPoint, 3 * sizeof(float), focal);
}

void MainWindow::on_rdoAlphaBlending_clicked()
{
    if (ui->rdoAlphaBlending->isChecked())
    {
        this->sendMessage(CompositingModeAlphaBlend, 0, nullptr);
    }
}

void MainWindow::on_rdoMIP_clicked()
{
    if (ui->rdoMIP->isChecked())
    {
        this->sendMessage(CompositingModeMIP, 0, nullptr);
    }
}

void MainWindow::on_rdoIsoSurface_clicked()
{
    if (ui->rdoIsoSurface->isChecked())
    {
        this->sendMessage(CompositingModeIsoSurface, 0, nullptr);
    }
}

void MainWindow::on_sldIsoValue_valueChanged(int value)
{
    float iso_value = static_cast<float>(value) * 0.01f;
    this->sendMessage(IsoSurfaceValue, sizeof(float), &iso_value);

    std::cout << "ISO: " << iso_value << std::endl;
}

void MainWindow::on_btnWritePng_clicked()
{
    m_write_png_images = !m_write_png_images;

    if (m_write_png_images)
    {
        this->sendMessage(PngWriterOn, 0, nullptr);
        ui->btnWritePng->setText("On");
    }
    else
    {
        this->sendMessage(PngWriterOff, 0, nullptr);
        ui->btnWritePng->setText("Off");
    }
}
