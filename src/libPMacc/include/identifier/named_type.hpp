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

/* No namespace is needed because we only have defines*/

/** define a spezial identifier with name, type and a default value
 * @param in_type type of the value
 * @param name name of identifier
 * 
 * The created identifier has the folowing options:
 *          getDefaultValue() - return the defualt value
 *          getName()         - return the name of the identifier
 *          ::type            - get type of the value 
 * 
 * e.g. named_type(float,length)
 *      typedef length::type value_type; // is float
 *      printf("Identifier name: %s",length::getName()); //print Identifier name: length
 * 
 * to create a instance of this value_identifier you can use:
 *      length();   or length_
 * 
 */
#define named_type(in_type,name,...)                                           \
        identifier(name,                                                       \
        typedef name ThisType;                                                 \
        typedef in_type type;                                                  \
        static HDINLINE char* getName()                                        \
        {                                                                      \
                return #name;                                                  \
        }                                                                      \
        __VA_ARGS__                                                            \
    )
