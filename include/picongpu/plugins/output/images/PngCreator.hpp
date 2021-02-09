/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#pragma once

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

#include <pmacc/memory/boxes/DataBox.hpp>
#include "picongpu/plugins/output/header/MessageHeader.hpp"

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>


namespace picongpu
{
    using namespace pmacc;


    struct PngCreator
    {
        PngCreator(std::string name, std::string folder)
            : m_name(folder + "/" + name)
            , m_folder(folder)
            , m_createFolder(true)
            , m_isThreadActive(false)
        {
        }

        static std::string getName()
        {
            return std::string("png");
        }

        /** block until all shared resource are free
         *
         * take care that all resources used by `operator()`
         * can safely used without conflicts
         */
        void join()
        {
            if(m_isThreadActive)
            {
                workerThread.join();
                m_isThreadActive = false;
            }
        }

        ~PngCreator()
        {
            if(m_isThreadActive)
            {
                workerThread.join();
                m_isThreadActive = false;
            }
        }

        PngCreator(const PngCreator& other)
        {
            m_name = other.m_name;
            m_folder = other.m_folder;
            m_createFolder = other.m_createFolder;
            m_isThreadActive = false;
        }

        /** create image
         *
         * @param data input data for png
         *             this object must be alive until destructor
         *             of `PngCreator` or method `join()` is called
         * @param size size of data
         * @param header meta information about the simulation
         */
        template<class Box>
        void operator()(const Box data, const MessageHeader::Size2D size, const MessageHeader header)
        {
            if(m_isThreadActive)
            {
                workerThread.join();
            }
            m_isThreadActive = true;
            workerThread = std::thread(&PngCreator::createImage<Box>, this, data, size, header);
        }

    private:
        template<class Box>
        void createImage(const Box data, const MessageHeader::Size2D size, const MessageHeader header);

        std::string m_name;
        std::string m_folder;
        bool m_createFolder;
        std::thread workerThread;
        /* status whether a thread is currently active */
        bool m_isThreadActive;
    };

} /* namespace picongpu */

#include "picongpu/plugins/output/images/PngCreator.tpp"
