/**
 * Copyright 2013-2016 Benjamin Schneider, Axel Huebl, Rene Widera
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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <mutex>

#include "rivlib/rivlib.h"

#include <QMainWindow>
#include <QPropertyAnimation>
#include "glwidget.h"
#include "simulationgallery.h"
#include "transferfunctions.h"

#define MEASURE_TIME_CPUCLOCK 1

namespace Ui {
class MainWindow;
}

using namespace eu_vicci;

class MainWindow : public QMainWindow,
                   public rivlib::image_stream_connection::listener,
                   public rivlib::control_connection::listener
{
    Q_OBJECT

public:

    explicit MainWindow(QWidget * parent = 0);
    ~MainWindow();

    void initInfoQuery(std::string ip, int m_infoport) { m_simulationGallery->startQuery(ip, m_infoport); }

    /** Implementation of image_stream_connection::listener */
    virtual void on_image_data(rivlib::image_stream_connection::ptr comm, uint32_t width, uint32_t height, const void * rgbpix) throw();

    virtual void on_error(rivlib::image_stream_connection::ptr comm, const char * msg) throw();

    virtual void on_connected(rivlib::image_stream_connection::ptr comm) throw();

    virtual void on_disconnected(rivlib::image_stream_connection::ptr comm) throw();

    /** Implementation of control_connection::listener */
    virtual void on_connected(rivlib::control_connection::ptr comm) throw();

    virtual void on_disconnected(rivlib::control_connection::ptr comm) throw();

    virtual void on_error(rivlib::control_connection::ptr comm, const char *msg) throw();

    virtual void on_msg(rivlib::control_connection::ptr comm, unsigned int id, unsigned int size, const void *data) throw();

public slots:

    void triggerDataSidebar();
    void triggerTransferfuncSidebar();
    void triggerSettingsSidebar();
    void triggerInfoBar();
    void triggerClipBar();

    void triggerUi();

    void hideAllSidebars();

    void triggerSimulationGallery();

    void changeBackgroundcolor();

    void connectToURI(QString uri);

    void received_datasource(QString source);
    void received_timestep(int step);
    void received_fps(float fps);
    void received_rfps(float fps);

    void received_numGPUs(int numGPUs);
    void received_numCells(double numCells);
    void received_numParticles(double numParticles);

    void sendMessage(unsigned int id, unsigned int size, const void * data);

    void transferFunctionA_changed();
    void transferFunctionB_changed();

    void playPauseSim();

    void clippingChanged();

    void refresh_glWidget();

signals:

    void on_received_timestep(int step);
    void on_received_fps(float fps);
    void on_received_rfps(float fps);
    void on_received_datasource(QString source);

    void on_received_numGPUs(int numGPUs);
    void on_received_numCells(double numCells);
    void on_received_numParticles(double numParticles);

    void on_transferFunctionA_changed();
    void on_transferFunctionB_changed();

    void on_correct_sidebar_positions();

protected:

    void resizeEvent(QResizeEvent * re);

    void reset_connection();

private slots:
    void on_cmbDatasourceA_currentIndexChanged(const QString &arg1);

    void on_cmbDatasourceB_currentIndexChanged(const QString &arg1);

    void on_rdoRedGreenA_toggled(bool checked);

    void on_rdoTempA_toggled(bool checked);

    void on_rdo2HueA_toggled(bool checked);

    void on_rdoRedGreenB_toggled(bool checked);

    void on_rdoTempB_toggled(bool checked);

    void on_rdo2HueB_toggled(bool checked);

    void on_sldOffsetXTFA_valueChanged(int value);

    void on_sldSlopeTFA_valueChanged(int value);

    void on_sldOffsetYTFA_valueChanged(int value);

    void on_sldOffsetXTFB_valueChanged(int value);

    void on_sldSlopeTFB_valueChanged(int value);

    void on_sldOffsetYTFB_valueChanged(int value);

    void on_sldSourceWeighting_valueChanged(int value);

    void on_sldOffsetXTFA_sliderReleased();

    void correct_sidebar_positions();

    void on_pushButton_clicked();

    void on_rdoAlphaBlending_clicked();

    void on_rdoMIP_clicked();

    void on_rdoIsoSurface_clicked();

    void on_sldIsoValue_valueChanged(int value);

    void on_btnWritePng_clicked();

private:

    Ui::MainWindow * ui;

    bool m_uiVisible;

    /** Simulation gallery to choose to which simualtion the user wants to connect */
    SimulationGallery * m_simulationGallery;
    bool m_simulationGalleryOnScreen;

    /** RIVLib */
    rivlib::image_stream_connection::ptr m_imgStream;
    rivlib::control_connection::ptr m_controlConn;

    /** Animation control properties */
    bool m_dataSidebarOnScreen;
    bool m_transferfuncSidebarOnScreen;
    bool m_settingsSidebarOnScreen;
    bool m_infoBarOnScreen;
    bool m_clipBarOnScreen;

    QPropertyAnimation * m_aniDataSidebar;
    QPropertyAnimation * m_aniTransferfuncSidebar;
    QPropertyAnimation * m_aniSettingsSidebar;
    QPropertyAnimation * m_aniInfoBar;
    QPropertyAnimation * m_aniClipBar;

    int LEFT_X_SHOWN;
    int LEFT_X_HIDDEN;

    int RIGHT_X_SHOWN;
    int RIGHT_X_HIDDEN;

    /** Backgroundcolor */
    float m_backgroundColor[3];

    /** Current Transferfunctions */
    ITransferFunction * m_currentTFA;
    ITransferFunction * m_currentTFB;

    /** Opacity Weighting of Datasource A and B. */
    float m_weighting;

    /** simulation control */
    bool m_isSimRunning;

    /** The Clipping box defined by six floats min x,y,z and max x,y,z. */
    float m_clip[6];

    /** Write Png images to simulation output folder. */
    bool m_write_png_images;

#if (MEASURE_TIME_CPUCLOCK == 1)
    struct timespec m_start, m_stop;
#endif
};

#endif // MAINWINDOW_H
