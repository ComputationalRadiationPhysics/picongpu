/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <mpi.h>

namespace pmacc
{
    namespace mpi
    {
        template<class Functor>
        MPI_Op getMPI_Op();
    } // namespace mpi
} // namespace pmacc
