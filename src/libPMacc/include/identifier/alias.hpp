/**
 * Copyright 2013-2014 Rene Widera, Felix Schmitt
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

namespace PMacc
{
identifier(pmacc_void);
identifier(pmacc_isAlias);
} //namespace PMacc

/*define special makros for creating classes which are only used as identifer*/
#define PMACC_alias(name,id)                                                   \
    namespace PMACC_JOIN(placeholder_definition,id) {                          \
        template<typename T_Type=PMacc::pmacc_void,typename T_IsAlias=PMacc::pmacc_isAlias> \
        struct name:public T_Type                                                   \
        {                                                                      \
            typedef T_Type ThisType;                                                \
            static HDINLINE char* getName()                                    \
            {                                                                  \
                     return #name;                                             \
            }                                                                  \
        };                                                                     \
    }                                                                          \
    using namespace PMACC_JOIN(placeholder_definition,id);                     \
    namespace PMACC_JOIN(host_placeholder,id){                                 \
        PMACC_JOIN(placeholder_definition,id)::name<> PMACC_JOIN(name,_);      \
    }                                                                          \
    namespace PMACC_JOIN(device_placeholder,id){                               \
        __constant__ PMACC_JOIN(placeholder_definition,id)::name<> PMACC_JOIN(name,_); \
    }                                                                          \
    PMACC_PLACEHOLDER(id);


/** create an alias
 *
 * an alias is a unspecialized type of a identifier or a value_identifier
 *
 * @param name name of alias
 *
 * example: alias(aliasName); //create type varname
 *
 * to specialized an alias do: aliasName<valueIdentifierName>
 * to create a instance of this alias you can use:
 *      aliasName();   or aliasName
 *
 * get type which is represented by the alias
 *      typedef typename name::ThisType type;
 */
#define alias(name) PMACC_alias(name,__COUNTER__)
