/* Copyright 2013-2021 Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cupla.hpp>

#include <iostream>
#include <stdexcept>


/**
 * Print a cupla error message including file/line info to stderr
 */
#define PMACC_PRINT_CUPLA_ERROR(msg)                                                                                  \
    std::cerr << "[cupla] Error: <" << __FILE__ << ">:" << __LINE__ << " " << msg << std::endl

/**
 * Print a cupla error message including file/line info to stderr and raises an exception
 */
#define PMACC_PRINT_CUPLA_ERROR_AND_THROW(cuplaError, msg)                                                            \
    PMACC_PRINT_CUPLA_ERROR(msg);                                                                                     \
    throw std::runtime_error(std::string("[cupla] Error: ") + std::string(cuplaGetErrorString(cuplaError)))

/**
 * Captures CUDA errors and prints messages to stdout, including line number and file.
 *
 * @param cmd command with cuplaError_t return value to check
 */
#define CUDA_CHECK(cmd)                                                                                               \
    {                                                                                                                 \
        cuplaError_t error = cmd;                                                                                     \
        if(error != cuplaSuccess)                                                                                     \
        {                                                                                                             \
            PMACC_PRINT_CUPLA_ERROR_AND_THROW(error, "");                                                             \
        }                                                                                                             \
    }

/** Capture error, report and throw
 *
 * This macro is only used when PMACC_SYNC_KERNEL == 1 to wrap all
 * kernel calls. Since alpaka may throw inside cmd, everything is
 * wrapped up in another try-catch level.
 *
 * This macro will always throw in case of an error, either by
 * producing a new exception or propagating an existing one
 */
#define CUDA_CHECK_MSG(cmd, msg)                                                                                      \
    {                                                                                                                 \
        try                                                                                                           \
        {                                                                                                             \
            cuplaError_t error = cmd;                                                                                 \
            if(error != cuplaSuccess)                                                                                 \
            {                                                                                                         \
                PMACC_PRINT_CUPLA_ERROR_AND_THROW(error, msg);                                                        \
            }                                                                                                         \
        }                                                                                                             \
        catch(...)                                                                                                    \
        {                                                                                                             \
            PMACC_PRINT_CUPLA_ERROR(msg);                                                                             \
            throw;                                                                                                    \
        }                                                                                                             \
    }

#define CUDA_CHECK_NO_EXCEPT(cmd)                                                                                     \
    {                                                                                                                 \
        cuplaError_t error = cmd;                                                                                     \
        if(error != cuplaSuccess)                                                                                     \
        {                                                                                                             \
            PMACC_PRINT_CUPLA_ERROR("");                                                                              \
        }                                                                                                             \
    }
