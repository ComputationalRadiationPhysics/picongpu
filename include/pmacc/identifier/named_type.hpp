/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/identifier/identifier.hpp"
#include "pmacc/types.hpp"

#include <string>

/* No namespace is needed because we only have defines*/

/** define a unique identifier with name, type and a default value
 * @param in_type type of the value
 * @param name name of identifier
 *
 * The created identifier has the following options:
 *          getName()         - return the name of the identifier
 *          ::type            - get contained type
 *
 * e.g. named_type(float,length)
 *      typedef length::type value_type; // is float
 *      printf("Identifier name: %s",length::getName()); //print Identifier name: length
 *
 * to create a instance of this value_identifier you can use:
 *      length();   or length_
 *
 */
#define named_type(in_type, name, ...)                                                                                \
    identifier(name, typedef in_type type; static std::string getName() { return std::string(#name); } __VA_ARGS__)
