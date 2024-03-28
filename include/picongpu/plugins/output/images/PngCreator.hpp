/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/output/header/MessageHeader.hpp"

#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/types.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


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
         * @param imageVector 1D representation of the image
         *             this object must be alive until destructor
         *             of `PngCreator` or method `join()` is called
         * @param header meta information about the simulation
         */
        template<typename T_DataType>
        void operator()(std::shared_ptr<std::vector<T_DataType>> imageVector, const MessageHeader header)
        {
            if(m_isThreadActive)
            {
                workerThread.join();
            }
            m_isThreadActive = true;
            workerThread = std::thread(&PngCreator::createImage<T_DataType>, this, imageVector, header);
        }

    private:
        template<typename T_DataType>
        void createImage(std::shared_ptr<std::vector<T_DataType>> imageBuffer, const MessageHeader header);

        std::string m_name;
        std::string m_folder;
        bool m_createFolder;
        std::thread workerThread;
        /* status whether a thread is currently active */
        bool m_isThreadActive;
    };

} /* namespace picongpu */

#include "picongpu/plugins/output/images/PngCreator.tpp"
