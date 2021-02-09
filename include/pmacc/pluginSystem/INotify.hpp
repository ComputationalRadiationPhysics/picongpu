/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Axel Huebl,
 *                     Richard Pausch
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

namespace pmacc
{
    /*
     * INotify interface.
     */
    class INotify
    {
    protected:
        uint32_t lastNotify;

    public:
        INotify() : lastNotify(0)
        {
        }

        virtual ~INotify()
        {
        }

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
