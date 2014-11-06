/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CURSOR_EMPTYNAVIGATOR_HPP
#define CURSOR_EMPTYNAVIGATOR_HPP

namespace PMacc
{
namespace cursor
{

struct EmptyNavigator
{
    template<typename Marker, typename Jump>
    HDINLINE
    Marker operator()(const Marker& marker, Jump) const
    {
        return marker;
    }
};

} // cursor
} // PMacc

#endif // CURSOR_EMPTYNAVIGATOR_HPP
