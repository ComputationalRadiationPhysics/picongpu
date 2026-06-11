/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/debug/abortWithError.hpp"

#include <cstdlib>

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

namespace pmacc
{
    /** verify expression with message on device
     *
     * Same behavior as PMACC_DEVICE_ASSERT_MSG but the expression is always evaluated.
     * The function is constexpr to avoid using function host device attributes.
     *
     * @param isValid if false code execution will stop with an error, else nothing will be executed
     * @param printfArgs arguments forwarded to printf
     */
    template<typename... T_PrintfArgs>
    constexpr void device_verify_msg(bool isValid, T_PrintfArgs&&... printfArgs)
    {
        if(!isValid)
        {
            printf(std::forward<T_PrintfArgs>(printfArgs)...);
#if (PMACC_DEVICE_COMPILE == 1)
#    if BOOST_COMP_HIP
            __builtin_trap();
#    elif BOOST_LANG_CUDA
            __trap();
#    endif
#else
            throw std::runtime_error("verify_msg_device");
#endif
        }
    }
} // namespace pmacc

/** verify expression with message on device
 *
 * Same behavior as PMACC_DEVICE_ASSERT_MSG but the expression is always evaluated.
 *
 * @ATTENTION Using this function can increase the register footprint.
 *
 * @param expr expression to be evaluated
 * @param ... arguments forwarded to printf
 */
#define PMACC_DEVICE_VERIFY_MSG(expr, ...) pmacc::device_verify_msg(!!(expr), __VA_ARGS__)
