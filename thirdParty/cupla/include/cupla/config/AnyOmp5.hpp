/* Copyright 2019 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include <alpaka/standalone/AnyOmp5.hpp>

#ifndef CUPLA_HEADER_ONLY
#    define CUPLA_HEADER_ONLY 1
#endif

#if(CUPLA_HEADER_ONLY == 1)
#    define CUPLA_HEADER_ONLY_FUNC_SPEC inline
#endif

#if(CUPLA_HEADER_ONLY == 1)
#    include "cupla/../../src/common.cpp"
#    include "cupla/../../src/device.cpp"
#    include "cupla/../../src/event.cpp"
#    include "cupla/../../src/manager/Driver.cpp"
#    include "cupla/../../src/memory.cpp"
#    include "cupla/../../src/stream.cpp"
#endif

#include "cupla.hpp"
