/**
 * Copyright 2013-2016 Benjamin Schneider
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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <mutex>

#include <QtOpenGL>
#include <QGLWidget>
#include <QKeyEvent>
#include "datasourcesidebar.h"
#include "transferfuncsidebar.h"
#include "camera.h"


class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    explicit GLWidget(QWidget *parent = 0);
    ~GLWidget();

    void update_image(uint width, uint height, const void * pixel_data);

    void set_background_color(float r, float g, float b);

    void set_simulation_area(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax);

    float3 get_simulation_center();

    Camera * getCamera() { return m_camera; }

signals:

    /**
     * @brief send_message Signals that a message should be send to the server.
     * @param msg The message as a string.
     */
    void send_message(unsigned int id, unsigned int size, const void * data);

    void trigger_dataSidebar();
    void trigger_transferfuncSidebar();
    void trigger_settingsSidebar();
    void trigger_infoBar();
    void trigger_clipBar();

    void trigger_ui();

    void play_pause_sim();

    void show_simulationGallery();

protected:

    /**
     * @brief initializeGL Initialize OpenGL for rendering.
     */
    void initializeGL();

    /**
     * @brief paintGL Paint the received image.
     */
    void paintGL();

    /**
     * @brief resizeGL Handle resizing of the widget.
     * @param w New width.
     * @param h New height.
     */
    void resizeGL(int w, int h);

    /**
     * Handle mouse (wheel) events to control camera position.
     */
    void mousePressEvent(QMouseEvent * me);

    //void mouseReleaseEvent(QMouseEvent * me);

    void mouseMoveEvent(QMouseEvent * me);

    /**
     * @brief wheelEvent Use mouse wheel to move camera towards or away from focal point.
     * @param we The event indicating the rotation of the mouse wheel.
     */
    void wheelEvent(QWheelEvent * we);

    void keyPressEvent(QKeyEvent * ke);

private:

    Camera * m_camera;

    float m_background_color[3];

    /** OpenGL rendering stuff */
    GLuint m_texture;
    int m_texWidth;
    int m_texHeight;

    /** storage for received image */
    unsigned char * m_imgBuffer;
    int m_inImgWidth;
    int m_inImgHeight;

    /** lock for synchronizing reads and writes to the image buffer */
    std::mutex m_imgLock;

    /** stores last mouse position to compute movement delta */
    int prev_mx, prev_my;

    /** The bounds of the visible simulation area. */
    float m_simulationArea[6];

    /** Helper method to send Camera Position and Focal Point */
    void send_camera_position_focus();
};

#endif // GLWIDGET_H
