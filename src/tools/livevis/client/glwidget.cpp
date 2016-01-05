/**
 * Copyright 2013-2016 Benjamin Schneider, Rene Widera
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

#include <iostream>
#include <GL/glu.h>

#include "message_ids.hpp"
#include "glwidget.h"

GLWidget::GLWidget(QWidget *parent)
  : QGLWidget(parent),
    m_texture(0),
    m_texWidth(0),
    m_texHeight(0),
    m_imgBuffer(nullptr),
    m_inImgWidth(0),
    m_inImgHeight(0)
{
    m_camera = new Camera(float3(1000,1000,1000), float3(0,0,0));
    m_camera->setUp(float3(1,0,0));
}

GLWidget::~GLWidget()
{
    ::glDeleteTextures(1, &m_texture);

    if (m_imgBuffer)
    {
        delete [] m_imgBuffer;
        m_imgBuffer = nullptr;
    }
}

void GLWidget::update_image(uint width, uint height, const void * pixel_data)
{
    if ((width <= 0) || (height <= 0))
        return;

    m_imgLock.lock();

    if (m_imgBuffer == nullptr)
    {
        m_imgBuffer = new unsigned char[width * height * 3];
    }

    m_inImgWidth = width;
    m_inImgHeight = height;

    ::memcpy(m_imgBuffer, pixel_data, width * height * 3);

    m_imgLock.unlock();

    std::cout << "[GLWidget] Image updated!" << std::endl;
}

void GLWidget::set_background_color(float r, float g, float b)
{
    m_background_color[0] = r;
    m_background_color[1] = g;
    m_background_color[2] = b;

    ::glClearColor(r, g, b, 1.0f);
}

void GLWidget::set_simulation_area(float xmin, float ymin, float zmin, float xmax, float ymax, float zmax)
{
    m_simulationArea[0] = xmin;
    m_simulationArea[1] = ymin;
    m_simulationArea[2] = zmin;
    m_simulationArea[3] = xmax;
    m_simulationArea[4] = ymax;
    m_simulationArea[5] = zmax;
}

float3 GLWidget::get_simulation_center()
{
    float3 center;
    center.x = m_simulationArea[0] + 0.5f * (m_simulationArea[3] - m_simulationArea[0]);
    center.y = m_simulationArea[1] + 0.5f * (m_simulationArea[4] - m_simulationArea[1]);
    center.z = m_simulationArea[2] + 0.5f * (m_simulationArea[5] - m_simulationArea[2]);

    return center;
}

void GLWidget::initializeGL()
{
    ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    ::glGenTextures(1, &m_texture);

    ::glEnable(GL_TEXTURE_2D);
    ::glBindTexture(GL_TEXTURE_2D, m_texture);
    unsigned char demoTex[] = {
            0, 0, 0,
            255, 0, 0,
            0, 255, 0,
            255, 255, 255 };
    ::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, (void*)demoTex);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisable(GL_TEXTURE_2D);

    m_texWidth = 2;
    m_texHeight = 2;
}

void GLWidget::resizeGL(int w, int h)
{
    ::glViewport(0, 0, w, h);
}

void GLWidget::paintGL()
{
    /// draw received simulation image
    ::glEnable(GL_TEXTURE_2D);

    // read in new pixel data - lock pixel buffer
    m_imgLock.lock();
    if ((m_inImgWidth > 0) && (m_inImgHeight > 0))
    {
        ::glBindTexture(GL_TEXTURE_2D, m_texture);
        ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        ::glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_inImgWidth, m_inImgHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, (void*)m_imgBuffer);
        m_texWidth = m_inImgWidth;
        m_texHeight = m_inImgHeight;
        ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        ::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        ::glBindTexture(GL_TEXTURE_2D, 0);

        m_inImgWidth = 0;
        m_inImgHeight = 0;
    }
    m_imgLock.unlock();

    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();

    double ww, wh;
    if (this->width() < this->height())
    {
        double s = static_cast<double>(this->width()) / static_cast<double>(this->height());
        ww = 1.0;
        wh = 1.0 / s;
        ::glScaled(1.0 / s, 1, 1.0);
    }
    else
    {
        double s = static_cast<double>(this->height()) / static_cast<double>(this->width());
        ww = 1.0 / s;
        wh = 1.0;
        ::glScaled(1, 1.0/s, 1.0);
    }

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    //::glScaled(0.95, 0.95, 1.0);

    ::glBindTexture(GL_TEXTURE_2D, m_texture);

    double txmin, txmax, tymin, tymax;
    double iw, ih;

    iw = 1.0;
    ih = static_cast<double>(this->m_texHeight) / static_cast<double>(this->m_texWidth);

    if ((iw / ih) > (ww / wh))
    {
        ::glScaled(ww / iw, ww / iw, 1.0);
    }
    else
    {
        ::glScaled(wh / ih, wh / ih, 1.0);
    }

    txmin = 0.5 / static_cast<double>(this->m_texWidth);
    txmax = 1.0 - txmin;
    tymin = 0.5 / static_cast<double>(this->m_texHeight);
    tymax = 1.0 - tymin;

    ::glColor4ub(255, 255, 255, 255);
    ::glBegin(GL_QUADS);
        ::glTexCoord2d(txmin, tymax); ::glVertex2d(-iw, -ih);
        ::glTexCoord2d(txmin, tymin); ::glVertex2d(-iw,  ih);
        ::glTexCoord2d(txmax, tymin); ::glVertex2d( iw,  ih);
        ::glTexCoord2d(txmax, tymax); ::glVertex2d( iw, -ih);
    ::glEnd();

    ::glBindTexture(GL_TEXTURE_2D, 0);
    ::glDisable(GL_TEXTURE_2D);

    /// draw bounding box
    /*::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();

    float3 eye = m_camera->getPosition();
    float3 target = m_camera->getFocalPoint();
    float3 up = m_camera->getUp();

    ::gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, up.x, up.y, up.z);

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::gluPerspective(45.0, 1.0, 1.0, 5000.0); // TODO adapt to camera fovy and aspect ratio

    ::glColor4f(1.0f - m_background_color[0], 1.0f - m_background_color[1], 1.0f - m_background_color[2], 1.0f);
    ::glBegin(GL_LINES);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[1], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[3], m_simulationArea[1], m_simulationArea[2]);

        ::glVertex3f(m_simulationArea[0], m_simulationArea[1], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[4], m_simulationArea[2]);

        ::glVertex3f(m_simulationArea[0], m_simulationArea[1], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[1], m_simulationArea[5]);

        ::glVertex3f(m_simulationArea[3], m_simulationArea[4], m_simulationArea[5]);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[4], m_simulationArea[5]);

        ::glVertex3f(m_simulationArea[3], m_simulationArea[4], m_simulationArea[5]);
        ::glVertex3f(m_simulationArea[3], m_simulationArea[1], m_simulationArea[5]);

        ::glVertex3f(m_simulationArea[3], m_simulationArea[4], m_simulationArea[5]);
        ::glVertex3f(m_simulationArea[3], m_simulationArea[4], m_simulationArea[2]);

        ::glVertex3f(m_simulationArea[0], m_simulationArea[4], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[4], m_simulationArea[5]);

        ::glVertex3f(m_simulationArea[0], m_simulationArea[1], m_simulationArea[5]);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[4], m_simulationArea[5]);

        ::glVertex3f(m_simulationArea[0], m_simulationArea[4], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[3], m_simulationArea[4], m_simulationArea[2]);

        ::glVertex3f(m_simulationArea[3], m_simulationArea[4], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[3], m_simulationArea[1], m_simulationArea[2]);

        ::glVertex3f(m_simulationArea[3], m_simulationArea[1], m_simulationArea[2]);
        ::glVertex3f(m_simulationArea[3], m_simulationArea[1], m_simulationArea[5]);

        ::glVertex3f(m_simulationArea[3], m_simulationArea[1], m_simulationArea[5]);
        ::glVertex3f(m_simulationArea[0], m_simulationArea[1], m_simulationArea[5]);

    ::glEnd();*/

    /// draw axes tripod in lower right corner

}

void GLWidget::wheelEvent(QWheelEvent * we)
{

    int delta = we->delta();

    float dolly;

    if (delta > 0)
    {
        dolly = 1.05f;
    }
    else if (delta < 0)
    {
        dolly = 0.95f;
    }
    else
    {
        dolly = 1.0f;
    }

    m_camera->dolly(dolly);

    float pos[3];
  //  pos[0] = m_camera->getPosition().x;
   // pos[1] = m_camera->getPosition().y;
   // pos[2] = m_camera->getPosition().z;

    pos[0] = 0.0f; //m_camera->getPosition().x;
    pos[1] = 0.0f;// m_camera->getPosition().y;
    pos[2] = dolly; //m_camera->getPosition().z;
    //emit send_message(CameraPosition, 3 * sizeof(float), pos);
    emit send_message(CameraOrbit, 3 * sizeof(float), pos);
}

void GLWidget::mousePressEvent(QMouseEvent * me)
{
    prev_mx = me->x();
    prev_my = me->y();
}

void GLWidget::mouseMoveEvent(QMouseEvent * me)
{

    int dx = me->x() - prev_mx;
    int dy = me->y() - prev_my;

    prev_mx = me->x();
    prev_my = me->y();

    static float factor = -0.25f;

    if (me->buttons() & Qt::LeftButton)
    {
        float yaw = static_cast<float>(dx) * factor;
        float pitch = static_cast<float>(dy) * factor;
        float roll = 0.0f;

        //  m_camera->orbitYawPitchRoll(yaw, pitch, roll);
        float pos[3];
        pos[0] = yaw; //m_camera->getPosition().x;
        pos[1] = pitch; // m_camera->getPosition().y;
        pos[2] = roll; //m_camera->getPosition().z;

        //emit send_message(CameraPosition, 3 * sizeof(float), pos);
        emit send_message(CameraOrbit, 3 * sizeof(float), pos);
    }

    if (me->buttons() & Qt::MidButton)
    {
        float yaw = static_cast<float>(dx) * factor; // * 0.25f;
        float pitch = static_cast<float>(dy) * factor; // * 0.25f;

        // m_camera->panYawPitch(yaw, pitch);

        float foc[2];
        foc[0] = yaw; //m_camera->getFocalPoint().x;
        foc[1] = pitch; //m_camera->getFocalPoint().y;
        //foc[2] = 0.0f; //m_camera->getFocalPoint().z;

        emit send_message(CameraPan, 2* sizeof(float), foc);
        //emit send_message(CameraFocalPoint, 3* sizeof(float), foc);
    }

    if (me->buttons() & Qt::RightButton)
    {
        float mx = -static_cast<float>(dx);
        float my = static_cast<float>(dy);

        m_camera->slide(float3(mx,my, 0.0f));

        float sli[3];
        sli[0] = mx;
        sli[1] = my;
        sli[2] = 0.0f;

        emit send_message(CameraSlide,3 * sizeof(float), sli);
    }

}

void GLWidget::keyPressEvent(QKeyEvent * ke)
{

    switch (ke->key())
    {

        case Qt::Key_W: {
            m_camera->slide(float3(0.0f, 0.0f, 0.5f));

            float sli[3];
            sli[0] = 0.f;
            sli[1] = 0.f;
            sli[2] = 0.5f;

            emit send_message(CameraSlide,3 * sizeof(float), sli);

        } break;

        case Qt::Key_S: {
            m_camera->slide(float3(0.0f, 0.0f, -0.5f));

            float sli[3];
            sli[0] = 0.f;
            sli[1] = 0.f;
            sli[2] = -0.5f;

            emit send_message(CameraSlide,3 * sizeof(float), sli);

        } break;

        case Qt::Key_A: {
            m_camera->slide(float3(-0.5f, 0.0f, 0.0f));

            float sli[3];
            sli[0] = -0.5f;
            sli[1] = 0.f;
            sli[2] = 0.f;

            emit send_message(CameraSlide,3 * sizeof(float), sli);

        } break;

        case Qt::Key_D: {
            m_camera->slide(float3(0.5f, 0.0f, 0.0f));

            float sli[3];
            sli[0] = 0.5f;
            sli[1] = 0.f;
            sli[2] = 0.f;

            emit send_message(CameraSlide,3 * sizeof(float), sli);

        } break;
/*
    case Qt::Key_Y:{
        emit send_message(CameraDefault, 0, nullptr);
        usleep(20000);

        float wei[1];
        wei[0] = 0.5f;
        emit send_message(Weighting, sizeof(float), wei);
        usleep(20000);

        float pos[3];
        pos[0] = m_camera->getPosition().x;
        pos[1] = m_camera->getPosition().y;
        pos[2] = m_camera->getPosition().z;

        emit send_message(CameraPosition, 3 * sizeof(float), pos);
        usleep(20000);

        emit send_message(SimPlay, 0 , nullptr);
        usleep(20000);

        float sli[3];
        for (int i = 0; i < 50; i++){
            m_camera->slide(float3(0.5f,0.f,0.f));

                    sli[0] = 0.5f;
                    sli[1] = 0.f;
                    sli[2] = 0.f;

                    emit send_message(CameraSlide,3 * sizeof(float), sli);
            usleep(20000);
        }
        for (int i = 0; i < 50; i++){
            m_camera->slide(float3(0.f,0.5f,0.f));

                        sli[0] = 0.f;
                        sli[1] = 0.5f;
                        sli[2] = 0.f;

                        emit send_message(CameraSlide,3 * sizeof(float), sli);
            usleep(20000);
        }

        for (int i = 0; i < 50; i++){
            m_camera->slide(float3(0.f,0.f,0.5f));

                        sli[0] = 0.f;
                        sli[1] = 0.f;
                        sli[2] = 0.5f;

                        emit send_message(CameraSlide,3 * sizeof(float), sli);
            usleep(20000);
        }

        for (int i = 0; i < 50; i++){
                        m_camera->slide(float3(-0.5f,-0.5f,-0.5f));

                        sli[0] = -0.5f;
                        sli[1] = -0.5f;
                        sli[2] = -0.5f;

                        emit send_message(CameraSlide,3 * sizeof(float), sli);
            usleep(20000);
                }

        emit send_message(SimPause, 0, nullptr);
        usleep(20000);

        emit send_message(CloseConnection, 0, nullptr);


    } break;
*/
        case Qt::Key_F1: {

            emit trigger_dataSidebar();
        } break;

        case Qt::Key_F2: {
            emit trigger_transferfuncSidebar();
        } break;

        case Qt::Key_F3: {
            emit trigger_settingsSidebar();
        } break;

        case Qt::Key_F4: {
            emit trigger_infoBar();
        } break;

        case Qt::Key_F5: {
            emit trigger_clipBar();
        } break;

        case Qt::Key_F11: {
            emit trigger_ui();
        } break;

        case Qt::Key_F12: {
            emit show_simulationGallery();
        } break;

        case Qt::Key_Space: {
            emit play_pause_sim();
        } break;
    }

   // this->send_camera_position_focus();
}

void GLWidget::send_camera_position_focus()
{
    float pos[3];
    float focus[3];

    pos[0] = m_camera->getPosition().x;
    pos[1] = m_camera->getPosition().y;
    pos[2] = m_camera->getPosition().z;

    focus[0] = m_camera->getFocalPoint().x;
    focus[1] = m_camera->getFocalPoint().y;
    focus[2] = m_camera->getFocalPoint().z;

    std::cout << "CameraPosition: " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
    std::cout << "CameraFocalPoint: " << focus[0] << " " << focus[1] << " " << focus[2] << std::endl;

    emit send_message(CameraPosition, 3 * sizeof(float), pos);
    emit send_message(CameraFocalPoint, 3 * sizeof(float), focus);
}
