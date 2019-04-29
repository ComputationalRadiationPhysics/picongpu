/* Copyright 2013-2019 Rene Widera, Benjamin Worpitz, Alexander Grund
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

#include "pmacc/types.hpp"
#include "pmacc/ppFunctions.hpp"

/* No namespace is needed because we only have defines*/

#ifdef __CUDA_ARCH__ //we are on gpu
#   define PMACC_PLACEHOLDER(id) using namespace PMACC_JOIN(device_placeholder,id)
#else
#   define PMACC_PLACEHOLDER(id) using namespace PMACC_JOIN(host_placeholder,id)
#endif

#ifdef __CUDACC__
#   define PMACC_identifier_CUDA(name,id)                                         \
        namespace PMACC_JOIN(device_placeholder,id){                               \
            __constant__ PMACC_JOIN(placeholder_definition,id)::name PMACC_JOIN(name,_); \
        }
#else
#   define PMACC_identifier_CUDA(name,id)
#endif

/*define special macros for creating classes which are only used as identifier*/
#define PMACC_identifier(name,id,...)                                          \
    namespace PMACC_JOIN(placeholder_definition,id) {                          \
        struct name{                                                           \
            __VA_ARGS__                                                        \
        };                                                                     \
    }                                                                          \
    using namespace PMACC_JOIN(placeholder_definition,id);                     \
    namespace PMACC_JOIN(host_placeholder,id){                                 \
        PMACC_JOIN(placeholder_definition,id)::name PMACC_JOIN(name,_);        \
    }                                                                          \
    PMACC_identifier_CUDA(name,id);                                            \
    PMACC_PLACEHOLDER(id);


/** create an identifier (identifier with arbitrary code as second parameter
 * !! second parameter is optional and can be any C++ code one can add inside a class
 *
 * example: identifier(varname); //create type varname
 * example: identifier(varname,typedef int type;); //create type varname,
 *          later its possible to use: typedef varname::type type;
 *
 * to create an instance of this identifier you can use:
 *      varname();   or varname_
 */
#define identifier(name,...) PMACC_identifier(name,__COUNTER__,__VA_ARGS__)
