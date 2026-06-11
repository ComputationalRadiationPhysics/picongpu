/*
 * SPDX-FileCopyrightText: Rene Widera, Pawel Ordyna
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/debug/abortWithError.hpp"

#include <cassert>

// disabled for no-debug mode or for the device compile path
#if defined(NDEBUG) || (PMACC_DEVICE_COMPILE == 1)

/* `(void)0` force a semicolon after the macro function */
#    define PMACC_ASSERT(expr) ((void) 0)

/* `(void)0` force a semicolon after the macro function */
#    define PMACC_ASSERT_MSG(expr, msg) ((void) 0)

#else

/** assert check (host side only)
 *
 * if `NDEBUG` is defined: macro expands to (void)0
 *
 * @param expr expression to be evaluated
 */
#    define PMACC_ASSERT(expr) (!!(expr)) ? ((void) 0) : pmacc::abortWithError(#expr, __FILE__, __LINE__)

/** assert check with message (host side only)
 *
 * if `NDEBUG` is defined: macro expands to (void)0
 *
 * @param expr expression to be evaluated
 * @param msg output message (of type `std::string`) which is printed if the
 *            expression is evaluated to false
 */
#    define PMACC_ASSERT_MSG(expr, msg) (!!(expr)) ? ((void) 0) : pmacc::abortWithError(#expr, __FILE__, __LINE__, msg)

#endif

// disabled for no-debug mode or for the host compile path
#if defined(NDEBUG) || (PMACC_DEVICE_COMPILE == 0)

/* `(void)0` force a semicolon after the macro function */
#    define PMACC_DEVICE_ASSERT(expr) ((void) 0)

// debug mode is disabled
/* `(void)0` force a semicolon after the macro function */
#    define PMACC_DEVICE_ASSERT_MSG(expr, ...) ((void) 0)

#else

/** assert check for kernels (device side)
 *
 * if `NDEBUG` is defined: macro expands to (void)0
 * @param expr expression to be evaluated
 */
#    define PMACC_DEVICE_ASSERT(expr) assert(expr)

/** assert check with message (device side)
 *
 * if `NDEBUG` is defined: macro expands to (void)0
 *
 * Beside the usual assert message an additional message is printed to stdout with `printf`.
 * Pass your `printf` arguments after the evaluated expression, for example to print some local variables:
 * @code{.cpp}
 * PMACC_DEVICE_ASSERT_MSG((x > 0), "x was %e, a was %e", x, a);
 * @endcode
 *
 * @param expr expression to be evaluated
 * @param ... parameters passed to printf
 */
#    define PMACC_DEVICE_ASSERT_MSG(expr, ...) (!!(expr)) ? ((void) 0) : (printf(__VA_ARGS__), assert(expr))
#endif
