/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include <pmacc/math/ConstVector.hpp>

#define CONST_VECTOR(type, dim, name, ...) PMACC_CONST_VECTOR(type, dim, name, __VA_ARGS__)
