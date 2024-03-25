/* Copyright 2013-2023 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <stdexcept>


/** Print a error message including file/line info to stderr
 */
#define PMACC_PRINT_ALPAKA_ERROR(msg)                                                                                 \
    do                                                                                                                \
    {                                                                                                                 \
        std::cerr << "[alpaka] Error: <" << __FILE__ << ">:" << __LINE__ << " " << msg << std::endl;                  \
    } while(false)

/** Print a error message including file/line info to stderr and raises an exception
 */
#define PMACC_PRINT_ALPAKA_ERROR_AND_THROW(msg)                                                                       \
    do                                                                                                                \
    {                                                                                                                 \
        PMACC_PRINT_ALPAKA_ERROR(msg);                                                                                \
        throw std::runtime_error(std::string("[alpaka] Error: ") + msg);                                              \
    } while(false)

/** Captures an expression in a catch throw block and prints messages to stdout, including line number and file.
 *
 * @param ... expression to capture in a catch throw block
 */
#define PMACC_CHECK_ALPAKA_CALL(...)                                                                                  \
    do                                                                                                                \
    {                                                                                                                 \
        try                                                                                                           \
        {                                                                                                             \
            __VA_ARGS__;                                                                                              \
        }                                                                                                             \
        catch(std::exception const& e)                                                                                \
        {                                                                                                             \
            PMACC_PRINT_ALPAKA_ERROR_AND_THROW(e.what());                                                             \
        }                                                                                                             \
    } while(false)

/** Capture error, report and throw
 *
 * This macro is only used when PMACC_SYNC_KERNEL == 1 to wrap all
 * kernel calls. Since alpaka may throw inside cmd, everything is
 * wrapped up in another try-catch level.
 *
 * This macro will always throw in case of an error, either by
 * producing a new exception or propagating an existing one
 */
#define PMACC_CHECK_ALPAKA_CALL_MSG(cmd, msg)                                                                         \
    do                                                                                                                \
    {                                                                                                                 \
        try                                                                                                           \
        {                                                                                                             \
            cmd;                                                                                                      \
        }                                                                                                             \
        catch(std::exception const& e)                                                                                \
        {                                                                                                             \
            PMACC_PRINT_ALPAKA_ERROR(e.what());                                                                       \
        }                                                                                                             \
    } while(false)

#define PMACC_CHECK_ALPAKA_CALL_NO_EXCEPT(...)                                                                        \
    do                                                                                                                \
    {                                                                                                                 \
        PMACC_CHECK_ALPAKA_CALL_MSG(__VA_ARGS__, "");                                                                 \
    } while(false)
