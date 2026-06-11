/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/alpakaHelper/acc.hpp"

#include <alpaka/alpaka.hpp>

#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace pmacc
{
    namespace manager
    {
        template<typename T_DeviceType>
        struct Device
        {
            using DeviceType = T_DeviceType;
            std::optional<DeviceType> m_device;

            int m_devIdx = -1;

            static Device& get()
            {
                static Device instance;
                return instance;
            }

            auto getPlatform() const
            {
                return alpaka::Platform<DeviceType>{};
            }

            auto device(int idx = 0) -> DeviceType&
            {
                if(m_devIdx != -1)
                {
                    if(m_devIdx == idx)
                        return *m_device;
                    else
                        throw std::runtime_error(
                            std::string("Device with id '") + std::to_string(m_devIdx)
                            + "' is already selected, changing the device is not allowed.");
                }
                else
                {
                    int const numDevices = count();
                    if(idx >= numDevices)
                    {
                        std::stringstream err;
                        err << "Unable to return device " << idx << ". There are only " << numDevices << " devices!";
                        throw std::runtime_error(err.str());
                    }

                    std::optional<DeviceType> dev;

                    auto const platform = getPlatform();
                    try
                    {
                        dev = std::make_optional<DeviceType>(alpaka::getDevByIdx(platform, idx));
                    }
                    catch(std::runtime_error const& e)
                    {
                        throw std::runtime_error(e.what());
                    }
                    m_device = std::move(dev);
                    m_devIdx = idx;
                    return *m_device;
                }
            }

            /**! reset the current device
             *
             * streams, memory and events on the current device must be
             * deleted at first by the user else the call will throw an error
             */
            void reset()
            {
                if(!m_device)
                {
                    std::stringstream err;
                    err << "device " << m_devIdx << " can not destroyed (was never created) ";
                    throw std::runtime_error(err.str());
                }
                else
                {
                    m_devIdx = -1;
                    m_device.reset();
                }
            }

            auto id() -> int
            {
                return m_devIdx;
            }

            auto current() -> DeviceType&
            {
                assert(m_device.has_value());
                return *m_device;
            }

            auto count() -> int
            {
                auto const platform = alpaka::Platform<DeviceType>{};
                return static_cast<int>(::alpaka::getDevCount(platform));
            }

        protected:
            Device()
            {
            }
        };

    } // namespace manager
} // namespace pmacc
