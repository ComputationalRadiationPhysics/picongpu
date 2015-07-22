/**
 * Copyright 2015  Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#pragma once

#include "particles/memory/dataTypes/Pointer.hpp"
#include "types.h"


namespace PMacc
{

template<typename T_Type = bmpl::_1>
struct PreviousFramePtr
{
    PMACC_ALIGN(previousFrame, Pointer<T_Type>);
};

template<typename T_Type = bmpl::_1>
struct NextFramePtr
{
    PMACC_ALIGN(nextFrame, Pointer<T_Type>);
};

} //namespace PMacc
