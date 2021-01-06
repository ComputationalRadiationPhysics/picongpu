/* Copyright 2015-2021 Rene Widera
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

/** echo given input */
#define PMACC_PP_ECHO(...) __VA_ARGS__

/** echo given input with delay */
#define PMACC_PP_DEFER_ECHO() PMACC_PP_ECHO

/** get the first element of a preprocessor pair */
#define PMACC_PP_FIRST(first, second) first

/** get the first element of a preprocessor pair with delay */
#define PMACC_PP_DEFER_FIRST() PMACC_PP_FIRST


/** get the second element of a preprocessor pair */
#define PMACC_PP_SECOND(first, second) second

/** get the second element of a preprocessor pair with delay */
#define PMACC_PP_DEFER_SECOND() PMACC_PP_SECOND

/** remove parentheses
 *
 * transform (...) to ...
 */
#define PMACC_PP_REMOVE_PAREN(...) PMACC_PP_DEFER_ECHO() __VA_ARGS__

/** remove parentheses with delay */
#define PMACC_PP_DEFER_REMOVE_PAREN() PMACC_PP_REMOVE_PAREN

/** call the given macro with the given argument.
 * can be used as a helper for expanding arguments that are lists
 */
#define PMACC_PP_CALL(macro, argument) macro argument
