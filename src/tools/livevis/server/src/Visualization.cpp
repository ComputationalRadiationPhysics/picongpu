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


#include <sys/socket.h>
// include <pngwriter.h> not implemented yet
#include <sstream>
#include <cstring>
#include <iomanip>

#include "../include/Visualization.h"
#include "../include/net/message_ids.hpp"

#define VERBOSITY_LEVEL 1

using namespace InSituVisualization;
using namespace picongpu::insituvolvis::net;

Visualization::Visualization(TCPStream * stream, rivlib::provider::ptr prov, std::string name, int image_width, int image_height)
    : m_prov(prov),
      m_stream(stream),
      m_name(name),
      m_thread(-1),
      m_isRunning(false),
      m_imageWidth(image_width),
      m_imageHeight(image_height)
{
    m_imageBuffer = new unsigned char[m_imageWidth * m_imageHeight * 3];

    /// set up RIVLib image data binding
    this->m_imgBinding.reset();

    if (this->m_prov)
    {
        this->m_prov->add_user_message_callback(&Visualization::onUserMessageProxy, this);
        this->m_prov->remove_all_data_bindings();
        this->m_imgBinding = rivlib::raw_image_data_binding::create(
                this->m_imageBuffer, m_imageWidth, m_imageHeight,
                rivlib::image_colour_type::rgb, rivlib::image_data_type::byte,
                rivlib::image_orientation::bottom_up, m_imageWidth * 3);

        this->m_prov->add_data_binding(this->m_imgBinding);
    }

#if VERBOSITY_LEVEL >= 1
    std::cout << "ImageSize " << m_imageWidth << " " << m_imageHeight << std::endl;
#endif
}


Visualization::~Visualization()
{
    m_isRunning = false;
    pthread_cancel(m_thread);

    delete m_stream;

    this->m_imgBinding.reset();
    if (this->m_prov) {
        this->m_prov->remove_user_message_callback(&Visualization::onUserMessageProxy, this);
        this->m_prov->shutdown();
        this->m_prov.reset();
    }

    if (this->m_imageBuffer != nullptr)
    {
        delete [] this->m_imageBuffer;
        this->m_imageBuffer = nullptr;
    }
}

void Visualization::runAsync()
{
    /// set up RIVLib image data binding
    this->m_imgBinding.reset();

    if (this->m_prov)
    {
        this->m_prov->remove_all_data_bindings();
        this->m_imgBinding = rivlib::raw_image_data_binding::create(
                this->m_imageBuffer, m_imageWidth, m_imageHeight,
                rivlib::image_colour_type::rgb, rivlib::image_data_type::byte,
                rivlib::image_orientation::bottom_up, m_imageWidth * 3);

        this->m_prov->add_data_binding(this->m_imgBinding);
    }

    /// start a thread for connected client
    m_isRunning = true;
    pthread_create(&m_thread, NULL, Visualization::runVisThread, (void*)this);
}

void Visualization::stop()
{
    m_isRunning = false;
    std::cout << "[SERVER](" << m_name << ") Finished." << std::endl;
}

void * Visualization::runVisThread(void * vis)
{
    Visualization * v = (Visualization*)vis;

    int buffer_size = v->getImageBufferSize();

    uint32_t id = 0;
    uint32_t len = 0;
    void * buffer = new char[buffer_size];

    /// receive messages which can contain image data (TCP) and send it to clients (RIV)
    /// or other information about the simulation
    while (v->isRunning())
    {
        /// receive message
        v->getStream()->receive(&id, buffer, &len);

        uint32_t timestep;
	float fps;
	float renderFps;

    int64_t numGPUs;
    int64_t numCells;
    int64_t numParticles;

        /// depending on what kind of header was sent treat data
        switch(id)
        {
            case RenderFPS: {
                renderFps = reinterpret_cast<float*>(buffer)[0];
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received RenderFPS " << renderFps << "." << std::endl;
#endif
                v->m_prov->broadcast_message(RIVLIB_USERMSG + RenderFPS, sizeof(float), (const char*)&renderFps);
                } break;
	    case FPS: {
                fps = reinterpret_cast<float*>(buffer)[0];
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received FPS " << fps << "." << std::endl;
#endif
                v->m_prov->broadcast_message(RIVLIB_USERMSG + FPS, sizeof(float), (const char*)&fps);
                } break;

            case NumGPUs: {
                 numGPUs = reinterpret_cast<int64_t*>(buffer)[0];
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received NumGPUs " << numGPUs << "." << std::endl;
#endif
                v->m_prov->broadcast_message(RIVLIB_USERMSG + NumGPUs, sizeof(int64_t), (const char*)&numGPUs);
                } break;
            case NumCells: {
                 numCells = reinterpret_cast<int64_t*>(buffer)[0];
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received NumCells " << numCells << "." << std::endl;
#endif
                v->m_prov->broadcast_message(RIVLIB_USERMSG + NumCells, sizeof(int64_t), (const char*)&numCells);
                } break;
            case NumParticles: {
                 numParticles = reinterpret_cast<int64_t*>(buffer)[0];
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received NumParticles " << numParticles << "." << std::endl;
#endif
                v->m_prov->broadcast_message(RIVLIB_USERMSG + NumParticles, sizeof(int64_t), (const char*)&numParticles);
                } break;

            case TimeStep: {
                timestep = reinterpret_cast<uint32_t*>(buffer)[0];
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received TimeStep " << timestep << "." << std::endl;
#endif
                v->m_prov->broadcast_message(RIVLIB_USERMSG + TimeStep, sizeof(uint32_t), (const char*)&timestep);
                } break;

            case AvailableDataSource: {

                char * c = new char[len + 1];
                std::memcpy(c, buffer, len);
                c[len] = 0;
                std::cout << "[SERVER](" << v->getName() << ") Received Available DataSource: " << c << std::endl;
                delete [] c;

                v->m_prov->broadcast_message(RIVLIB_USERMSG + id, len, (const char*)buffer);
            } break;

        case VisibleSimulationArea: {
        v->m_prov->broadcast_message(RIVLIB_USERMSG + id, len, (const char*)buffer);
        } break;

            case Image: {
#if VERBOSITY_LEVEL >= 2
                std::cout << "[SERVER](" << v->getName() << ") Received Image." << std::endl;
#endif
                v->processImage(len, buffer);
                } break;

            case CloseConnection: {
#if VERBOSITY_LEVEL >= 1
                std::cout << "[SERVER](" << v->getName() << ") Received Close Connection Message." << std::endl;
#endif
                v->stop();
                } break;

            case NoMessage: {
#if VERBOSITY_LEVEL >= 1
                std::cout << "[SERVER](" << v->getName() << ") Received no message!" << std::endl;
#endif
                //v->stop();
                } break;

            default:
#if VERBOSITY_LEVEL >= 1
                std::cout << "[SERVER](" << v->getName() << ") Received unknown Message." << std::endl;
#endif
                break;
        }

        std::memset(buffer, 0, buffer_size);
        id = 0;
        len = 0;
    }

    if (buffer != nullptr)
    {
        delete [] (char*)buffer;
        buffer = nullptr;
    }

    return nullptr;
}

void Visualization::processImage(uint32_t len, void * data)
{
    uint32_t image_size = this->getImageBufferSize();

    if (len != image_size)
    {
        std::cerr << "Image buffer size does not match size of received image data! (Buffer: " << image_size << " Byte)"
                  << " (Data: " << len << " Byte)" << std::endl;
        return;
    }

    /// wait for buffer to stop all operations
    if (m_imgBinding)
    {
        m_imgBinding->wait_async_data_completed();
    }

    /// if image buffer not yet created, allocate memory for RGB bytes
    if (m_imageBuffer == nullptr)
    {
        m_imageBuffer = new unsigned char[image_size];
        std::cout << "Created image buffer!" << std::endl;
    }

    /// copy recived data to image buffer
    std::memcpy(m_imageBuffer, data, len);

    /// publish new image data
    if (m_imgBinding)
    {
        m_imgBinding->async_data_available();
    }
#if VERBOSITY_LEVEL >= 2
    std::cout << "[SERVER](" << m_name << ") New Image Data available!" << std::endl;
#endif
}

void Visualization::onUserMessageProxy(unsigned int id, unsigned int size, const char * data, void * context)
{
    static_cast<Visualization*>(context)->onUserMessage(id, size, data);
}

void Visualization::onUserMessage(unsigned int id, unsigned int size, const char * data)
{
    /// remove RIVLIB user message ID offset and forward message to visualization
    m_stream->send(id - RIVLIB_USERMSG, data, size);

    /// DEBUG:
    //std::cout << "ID: " << id - RIVLIB_USERMSG << " Size: " << size << "." << std::endl;

    /*switch (id)
    {
        case RIVLIB_USERMSG + Weighting: {
                this->m_stream->send(Weighting, data, size);
            } break;
    }*/
}
