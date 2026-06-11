/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Axel Huebl, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include <cstdint>

namespace pmacc
{
    /*
     * INotify interface.
     */
    class INotify
    {
    protected:
        uint32_t lastNotify{0};

    public:
        INotify() = default;

        virtual ~INotify() = default;

        /** Notification callback
         *
         * For example Plugins can set their requested notification frequency at the
         * PluginConnector
         *
         * @param currentStep current simulation iteration step
         */
        virtual void notify(uint32_t currentStep) = 0;

        /** When was the plugin notified last?
         *
         * @return last notify time step
         */
        uint32_t getLastNotify() const
        {
            return lastNotify;
        }

        /** Remember last notification call
         *
         * @param currentStep current simulation iteration step
         */
        void setLastNotify(uint32_t currentStep)
        {
            lastNotify = currentStep;
        }
    };
} // namespace pmacc
