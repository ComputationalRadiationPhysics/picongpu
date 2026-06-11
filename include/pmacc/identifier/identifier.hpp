/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/ppFunctions.hpp"
#include "pmacc/types.hpp"

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
#define identifier(name, ...)                                                                                         \
    struct name                                                                                                       \
    {                                                                                                                 \
        __VA_ARGS__                                                                                                   \
    };                                                                                                                \
    constexpr name PMACC_JOIN(name, _)
