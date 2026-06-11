/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
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

        PngCreator(PngCreator const& other)
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
        void operator()(std::shared_ptr<std::vector<T_DataType>> imageVector, MessageHeader const header)
        {
            if(m_isThreadActive)
            {
                workerThread.join();
            }
            m_isThreadActive = true;
            workerThread = std::thread(&PngCreator::createImage<T_DataType>, this, imageVector, header);
        }

    private:
        /** Write image to disk
         *
         * @attention Only one MPI rank is allowed to call this method.
         */
        template<typename T_DataType>
        void createImage(std::shared_ptr<std::vector<T_DataType>> imageBuffer, MessageHeader const header);

        std::string m_name;
        std::string m_folder;
        bool m_createFolder;
        std::thread workerThread;
        /* status whether a thread is currently active */
        bool m_isThreadActive;
    };

} /* namespace picongpu */

#include "picongpu/plugins/output/images/PngCreator.tpp"
