/* Copyright 2015-2021 Heiko Burau
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
    namespace cursor
    {
        /**
         * Use this navigator to wrap a ndim-cursor into a 1D cursor.
         */
        template<int T_dim>
        class MapTo1DNavigator
        {
        public:
            static constexpr int dim = T_dim;

        private:
            math::Size_t<dim> shape;
            int pos;

            HDINLINE
            math::Int<dim> toNdim(int idx) const
            {
                math::Int<dim> result;
                int volume = 1;
                for(int i = 0; i < dim; i++)
                {
                    result[i] = (idx / volume) % this->shape[i];
                    volume *= this->shape[i];
                }
                return result;
            }

        public:
            /**
             * @param shape area to map the 1D index to.
             */
            HDINLINE
            MapTo1DNavigator(math::Size_t<dim> shape) : shape(shape), pos(0)
            {
            }

            template<typename Cursor>
            HDINLINE Cursor operator()(const Cursor& cursor, math::Int<1> jump)
            {
                math::Int<dim> ndstart = toNdim(this->pos);
                this->pos += jump.x();
                math::Int<dim> ndend = toNdim(this->pos);

                math::Int<dim> ndjump = ndend - ndstart;

                return cursor(ndjump);
            }
        };

    } // namespace cursor
} // namespace pmacc
