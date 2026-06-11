/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
