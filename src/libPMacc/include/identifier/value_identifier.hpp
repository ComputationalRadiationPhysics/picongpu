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

#pragma once

#include "types.h"
#include "identifier/identifier.hpp"
#include <string>

/* No namespace is needed because we only have defines*/

/** define a unique identifier with name, type and a default value
 * @param in_type type of the value
 * @param name name of identifier
 * @param in_default default value of in_type (can be a constructor of a class)
 *
 * The created identifier has the following options:
 *          getDefaultValue() - return the default value
 *          getName()         - return the name of the identifier
 *          ::type            - get type of the value
 *
 * e.g. value_identifier(float,length,0.0f)
 *      typedef length::type value_type; // is float
 *      value_type x= length::getDefault();  //set x to 0.f
 *      printf("Identifier name: %s",length::getName()); //print Identifier name: length
 *
 * to create a instance of this value_identifier you can use:
 *      length();   or length_
 *
 */
#define value_identifier(in_type,name,in_default)                              \
        identifier(name,                                                       \
        typedef name ThisType;                                                 \
        typedef in_type type;                                                  \
        static HDINLINE type getDefaultValue()                                 \
        {                                                                      \
                return in_default;                                             \
        }                                                                      \
        static std::string getName()                                           \
        {                                                                      \
                return std::string(#name);                                     \
        }                                                                      \
    )
