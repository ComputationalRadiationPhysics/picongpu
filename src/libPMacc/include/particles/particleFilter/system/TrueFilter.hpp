/**
 * Copyright 2013 Rene Widera
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


#ifndef TRUEFILTER_HPP
#define	TRUEFILTER_HPP

#include "types.h"
#include "particles/frame_types.hpp"
#include "particles/memory/frames/NullFrame.hpp"

namespace PMacc
{


    class TrueFilter
    {

    public:

        HDINLINE TrueFilter()
        {
        }

        HDINLINE virtual ~TrueFilter()
        {
        }

        template<class FRAME>
        HDINLINE bool operator()(FRAME&, lcellId_t)
        {
            return true;
        }
    };

} //namespace Frame

#endif	/* TRUEFILTER_HPP */

