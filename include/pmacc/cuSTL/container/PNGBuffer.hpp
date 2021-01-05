/* Copyright 2013-2021 Heiko Burau, Rene Widera
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

#include "pmacc/cuSTL/cursor/Cursor.hpp"
#include "pmacc/cuSTL/cursor/navigator/MultiIndexNavigator.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/math/vector/Float.hpp"
#include "pmacc/cuSTL/zone/SphericZone.hpp"
#include <pngwriter.h>
#include <string>

namespace pmacc
{
    namespace container
    {
        /** Think of a container being a PNG-image
         * offers only write-only access
         */
        class PNGBuffer
        {
        private:
            class Plotter
            {
            private:
                pngwriter& png;
                math::Int<2> pos;

            public:
                Plotter(pngwriter& png) : png(png)
                {
                }
                inline Plotter& operator=(const math::Float<3>& color)
                {
                    png.plot(pos.x() + 1, pos.y() + 1, (double) color.x(), (double) color.y(), (double) color.z());
                    return *this;
                }
                void setPos(const math::Int<2>& pos)
                {
                    this->pos = pos;
                }
            };
            struct Accessor
            {
                typedef Plotter& type;
                pngwriter& png;
                Plotter plotter;
                Accessor(pngwriter& png) : png(png), plotter(png)
                {
                }
                inline type operator()(math::Int<2>& index)
                {
                    plotter.setPos(index);
                    return this->plotter;
                }
            };
            pngwriter png;
            math::Size_t<2> size;

        public:
            typedef cursor::Cursor<PNGBuffer::Accessor, cursor::MultiIndexNavigator<2>, math::Int<2>> Cursor;

            /* constructor
             * \param x width of png image
             * \param y height of png image
             * \name name of png file
             */
            PNGBuffer(int x, int y, const std::string& name) : png(x, y, 0.0, name.data()), size(x, y)
            {
            }
            PNGBuffer(math::Size_t<2> size, const std::string& name)
                : png(size.x(), size.y(), 0.0, name.data())
                , size(size)
            {
            }
            ~PNGBuffer()
            {
                png.close();
            }

            /* get a cursor at the top left pixel
             * access via a Float<3> reference
             */
            inline Cursor origin()
            {
                return Cursor(Accessor(this->png), cursor::MultiIndexNavigator<2>(), math::Int<2>(0));
            }

            /* get a zone spanning the whole container */
            inline zone::SphericZone<2> zone() const
            {
                return zone::SphericZone<2>(this->size);
            }
        };

    } // namespace container
} // namespace pmacc
