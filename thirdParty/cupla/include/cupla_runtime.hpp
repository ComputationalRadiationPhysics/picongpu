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

#include <alpaka/alpaka.hpp>

#include "cupla/kernel.hpp"

#include "cupla/c/datatypes/cuplaArray.hpp"
#include "cupla/datatypes/dim3.hpp"
#include "cupla/datatypes/uint.hpp"
#include "cupla/c/datatypes/cuplaExtent.hpp"
#include "cupla/c/datatypes/cuplaPos.hpp"
#include "cupla/c/datatypes/cuplaMemcpy3DParms.hpp"
#include "cupla/c/datatypes/cuplaPitchedPtr.hpp"

#include "cupla/types.hpp"
#include "cupla_driver_types.hpp"

#include "cupla/api/common.hpp"
#include "cupla/api/device.hpp"
#include "cupla/api/stream.hpp"
#include "cupla/api/event.hpp"
#include "cupla/api/memory.hpp"
#include "cupla/manager/Driver.hpp"

namespace cupla
{
    const auto driver = manager::Driver::get();
}
