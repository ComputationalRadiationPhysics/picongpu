/* Copyright 2016-2021 Rene Widera
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

#include "pmacc/debug/abortWithError.hpp"

/** verify expression
 *
 * Same behavior as PMACC_ASSERT but the expression is always evaluated.
 *
 * @param expr expression to be evaluated
 */
#define PMACC_VERIFY(expr) (!!(expr)) ? ((void) 0) : pmacc::abortWithError(#expr, __FILE__, __LINE__)

/** verify expression with message
 *
 * Same behavior as PMACC_ASSERT_MSG but the expression is always evaluated.
 *
 * @param expr expression to be evaluated
 * @param msg output message (of type `std::string`) which is printed if the
 *            expression is evaluated to false
 */
#define PMACC_VERIFY_MSG(expr, msg) (!!(expr)) ? ((void) 0) : pmacc::abortWithError(#expr, __FILE__, __LINE__, msg)
