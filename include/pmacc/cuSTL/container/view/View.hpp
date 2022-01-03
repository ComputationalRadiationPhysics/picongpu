/* Copyright 2013-2022 Heiko Burau, Rene Widera
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
    namespace container
    {
        /** Represents a clipped area of its inherited container.
         *
         * View are not designed to do hard data copies.
         * Views don't take care of reference counters. So if the corresponding
         * container dies, all views become invalid.
         * Usual way to contruct a view goes with container.view(...);
         * @tparam Buffer Corresponding container type
         */
        template<typename Buffer>
        struct View : public Buffer
        {
            HINLINE View() = default;

            template<typename TBuffer>
            HINLINE View(const View<TBuffer>& other)
            {
                *this = other;
            }

            ~View() = default;

            template<typename TBuffer>
            HINLINE View& operator=(const View<TBuffer>& other)
            {
                this->sharedPtr = other.sharedPtr;
                this->shiftedPtr = other.get();
                this->_size = other._size;
                this->pitch = other.pitch;

                return *this;
            }

        private:
            // forbid view = container
            HINLINE Buffer& operator=(const Buffer& rhs);
        };


    } // namespace container
} // namespace pmacc
