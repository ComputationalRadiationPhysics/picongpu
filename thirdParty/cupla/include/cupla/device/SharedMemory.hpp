/* Copyright 2020 Rene Widera
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

#include <alpaka/alpaka.hpp>

#define sharedMem(ppName, ...)                                                 \
     __VA_ARGS__& ppName =                                                     \
        ::alpaka::declareSharedVar< __VA_ARGS__, __COUNTER__ >( acc )

#define sharedMemExtern(ppName, ...)                                           \
    __VA_ARGS__* ppName =                                                      \
        ::alpaka::getDynSharedMem< __VA_ARGS__ >( acc )
