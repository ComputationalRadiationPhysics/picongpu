/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/particles/memory/frames/NullFrame.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    class FalseFilter
    {
    public:
        FalseFilter()
        {
        }

        virtual ~FalseFilter()
        {
        }

        template<class FRAME>
        bool operator()(FRAME& frame, lcellId_t id)
        {
            return false;
        }
    };

} // namespace pmacc
