/*
 * SPDX-FileCopyrightText: Rene Widera, Wolfgang Hoenig, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include <mpi.h>

int const GridManagerRank = 0;

enum
{
    gridInitTag = 1,
    gridHostnameTag = 2,
    gridHostRankTag = 3,
    gridExitTag = 4,
    gridExchangeTag = 5
};

#define MPI_CHECK(cmd)                                                                                                \
    {                                                                                                                 \
        int error = cmd;                                                                                              \
        if(error != MPI_SUCCESS)                                                                                      \
        {                                                                                                             \
            std::cerr << "<" << __FILE__ << ">:" << __LINE__;                                                         \
            throw std::runtime_error(std::string("[MPI] Error"));                                                     \
        }                                                                                                             \
    }

#define MPI_CHECK_NO_EXCEPT(cmd)                                                                                      \
    {                                                                                                                 \
        int error = cmd;                                                                                              \
        if(error != MPI_SUCCESS)                                                                                      \
        {                                                                                                             \
            std::cerr << "[MPI] Error code " << error << " in <" << __FILE__ << ">:" << __LINE__;                     \
        }                                                                                                             \
    }
