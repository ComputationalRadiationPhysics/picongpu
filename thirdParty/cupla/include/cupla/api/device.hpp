/* Copyright 2016 Rene Widera
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

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"
#include "cupla_driver_types.hpp"

#include <alpaka/alpaka.hpp>

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
    cuplaError_t cuplaGetDeviceCount(int* count);

    cuplaError_t cuplaSetDevice(int idx);

    cuplaError_t cuplaGetDevice(int* deviceId);

    cuplaError_t cuplaDeviceReset();

    cuplaError_t cuplaDeviceSynchronize();

    cuplaError_t cuplaMemGetInfo(size_t* free, size_t* total);

} // namespace CUPLA_ACCELERATOR_NAMESPACE
