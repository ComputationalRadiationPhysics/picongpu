/**
 * Copyright 2016 Rene Widera
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

#include <system_error>

// emulated that cuda runtime is loaded
#ifndef __DRIVER_TYPES_H__
# define __DRIVER_TYPES_H__
#endif

enum cuplaMemcpyKind
{
  cuplaMemcpyHostToHost,
  cuplaMemcpyHostToDevice,
  cuplaMemcpyDeviceToHost,
  cuplaMemcpyDeviceToDevice
};

enum cuplaError
{
    cuplaSuccess = 0,
    cuplaErrorMemoryAllocation = 2,
    cuplaErrorInitializationError = 3,
    cuplaErrorInvalidDevice = 8,
    cuplaErrorNotReady = 34,
    cuplaErrorDeviceAlreadyInUse = 54
};

enum EventProp
{
    cuplaEventDisableTiming = 2
};

using cuplaError_t = enum cuplaError;


using cuplaStream_t = void*;

using cuplaEvent_t = void*;


/** error category for `cuplaError` */
struct CuplaErrorCode : public std::error_category
{
    virtual const char *name() const noexcept override { return "cuplaError"; }
    virtual std::string message(int ev) const override {
        switch(ev)
        {
            case cuplaSuccess:
                return "cuplaSuccess";
            case cuplaErrorMemoryAllocation:
                return "cuplaErrorMemoryAllocation";
            case cuplaErrorInitializationError:
                return "cuplaErrorInitializationError";
            case cuplaErrorNotReady:
                return "cuplaErrorNotReady";
            case cuplaErrorDeviceAlreadyInUse:
                return "cuplaErrorDeviceAlreadyInUse";
            default:
                return "not defined cuplaError";
        };
    }
};

namespace std
{

    template< >
    struct is_error_code_enum< cuplaError > : public true_type{};

} // namespace std

inline std::error_code make_error_code( const cuplaError result )
{
    return std::error_code( static_cast<int>(result), CuplaErrorCode() );
}


