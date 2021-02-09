/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace nvidia
    {
        namespace functors
        {
            struct Add
            {
                template<typename Dst, typename Src>
                HDINLINE void operator()(Dst& dst, const Src& src) const
                {
                    dst += src;
                }

                template<typename Dst, typename Src, typename T_Acc>
                HDINLINE void operator()(const T_Acc&, Dst& dst, const Src& src) const
                {
                    dst += src;
                }
            };
        } // namespace functors
    } // namespace nvidia
} // namespace pmacc

namespace pmacc
{
    namespace mpi
    {
        template<>
        HINLINE MPI_Op getMPI_Op<pmacc::nvidia::functors::Add>()
        {
            return MPI_SUM;
        }
    } // namespace mpi
} // namespace pmacc
