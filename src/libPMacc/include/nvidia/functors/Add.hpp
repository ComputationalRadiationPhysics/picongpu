/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef ADD_HPP
#define	ADD_HPP

#include "types.h"
#include "mpi/GetMPI_Op.hpp"

namespace PMacc
{
    namespace nvidia
    {
        namespace functors
        {

            struct Add
            {

                template<typename Dst, typename Src >
                        HDINLINE void operator()(Dst & dst, const Src & src) const
                {
                    dst += src;
                }
            };
        }
    }
}

namespace PMacc
{
    namespace mpi
    {

        template<>
        MPI_Op getMPI_Op<PMacc::nvidia::functors::Add>()
        {
            return MPI_SUM;
        }
    }
}

#endif	/* ADD_HPP */

