/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once


#include <pmacc/preprocessor/size.hpp>


#define PMACC_MIN(x, y) (((x) <= (y)) ? x : y)
#define PMACC_MAX(x, y) (((x) > (y)) ? x : y)


#define PMACC_JOIN_DO(x, y) x##y
#define PMACC_JOIN(x, y) PMACC_JOIN_DO(x, y)

#define PMACC_MAX_DO(what, x, y) (((x) > (y)) ? x what : y what)
#define PMACC_MIN_DO(what, x, y) (((x) < (y)) ? x what : y what)


#ifdef PMACC_PP_VARIADIC_SIZE
#    define PMACC_COUNT_ARGS_DEF(type, ...) (PMACC_PP_VARIADIC_SIZE(__VA_ARGS__))
#else
// A fallback implementation using compound literals, supported by some compilers
#    define PMACC_COUNT_ARGS_DEF(type, ...) (sizeof((type[]) {type{}, ##__VA_ARGS__}) / sizeof(type) - 1u)
#endif

/**
 * Returns number of args... arguments.
 *
 * @param type type of the arguments in ...
 * @param ... arguments
 */
#define PMACC_COUNT_ARGS(type, ...) PMACC_COUNT_ARGS_DEF(type, __VA_ARGS__)

/**
 * Check if ... has arguments or not
 *
 * Can only used if values of ... can be casted to int type
 *
 * @param ... arguments
 * @return false if no arguments are given, else true
 */
#define PMACC_HAS_ARGS(...) (PMACC_COUNT_ARGS(int, __VA_ARGS__) > 0)

/** round up to next higher pow 2 value
 *
 * - if value is pow2, value is returned
 * - maximal pow 2 value is 128
 * - negative values are not supported
 *
 * @param value integral number between [1,Inf]
 * @return next higher pow 2 value
 */
#define PMACC_ROUND_UP_NEXT_POW2(value)                                                                               \
    ((value) == 1                                                                                                     \
         ? 1                                                                                                          \
         : ((value) <= 2                                                                                              \
                ? 2                                                                                                   \
                : ((value) <= 4                                                                                       \
                       ? 4                                                                                            \
                       : ((value) <= 8 ? 8                                                                            \
                                       : ((value) <= 16 ? 16 : ((value) <= 32 ? 32 : ((value) <= 64 ? 64 : 128)))))))

/** Removes brackets from an macro function parameter
 *
 *  @code{.cpp}
 *  PMACC_REMOVE_BRACKETS (foo)
 *
 *  // will be transformed to:
 *  //   foo
 *  @endcode
 */
#define PMACC_REMOVE_BRACKETS(...) __VA_ARGS__
