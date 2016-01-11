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

#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <iostream>
#include <string>
#include <sys/socket.h>
#include <rivlib/rivlib.h>
#include "../include/net/tcp_stream.hpp"

namespace InSituVisualization
{

using namespace eu_vicci;
using namespace picongpu::insituvolvis::net;

/**
 * Represents a TCP-connected visualization sending images and
 * receiving control commands (e.g. camera position).
 */
class Visualization
{
public:

    explicit Visualization(TCPStream * stream, rivlib::provider::ptr prov, std::string name, int image_width, int image_height);

    virtual ~Visualization();

    /** Returns the name of the visualization. */
    const std::string& getName() const
    {
        return m_name;
    }

    /** Returns the provider of the visualization to which RIV clients can connect. */
    const rivlib::provider::ptr getProvider() const
    {
        return m_prov;
    }

    /**
     * Returns true if the thread for this visualization is running,
     * false if it is not yet running or has been stopped.
     */
    bool isRunning()
    {
        return m_isRunning;
    }

    /**
     * Returns a pointer to the buffer storing the image stream received
     * from the visualization. The format is RGB byte (byte == unsigned char).
     */
    unsigned char * getImageBuffer()
    {
        return m_imageBuffer;
    }

    /** Returns the size of the image buffer in bytes. Format is assumed to be RGB. */
    int getImageBufferSize()
    {
        return (m_imageWidth * m_imageHeight * 3);
    }

    /**
     * Starts a new thread for the just connected visualization and returns immediately.
     */
    void runAsync();

    /**
     * Stop the visualization thread. The server is shutting down the connection or the
     * visualization has disconnected.
     */
    void stop();

    TCPStream * getStream() { return m_stream; }

protected:

    /**
     * Creates a new pthread to runs the visualization connection in and hands over
     * "this" as parameter.
     */
    static void * runVisThread(void * vis);

    /**
     * Helper method to process received image data from the simulation.
     */
    void processImage(uint32_t len, void * data);

    /**
     * Proxy method to handle messages from the RIV client.
     */
    static void onUserMessageProxy(unsigned int id, unsigned int size, const char * data, void * context);

    /**
     * Method which really handles the RIV user message in the right context (i.e. on the correct object).
     */
    void onUserMessage(unsigned int id, unsigned int size, const char * data);

private:

    rivlib::provider::ptr m_prov;
    rivlib::raw_image_data_binding::ptr m_imgBinding;

    TCPStream * m_stream;

    /** Name of the connected simulation. */
    std::string m_name;

    /** The thread ID running this visualization on the server. */
    pthread_t m_thread;
    bool m_isRunning;

    /** Buffer to store the image received from the simulation. */
    unsigned char * m_imageBuffer;

    /** Size of the received image. */
    int m_imageWidth, m_imageHeight;

    /** Size of the simulation grid. */
    int m_gridX, m_gridY, m_gridZ;

    /** GPUs used in each dimension. */
    int m_gpusX, m_gpusY, m_gpusZ;

    /** Current time step of the simulation. */
    uint32_t m_currentStep;
};

} /* end of namespace InSituVisualization */

#endif // VISUALIZATION_H
