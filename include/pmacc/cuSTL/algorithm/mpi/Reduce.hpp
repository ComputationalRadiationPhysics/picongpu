/* Copyright 2013-2021 Heiko Burau
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

#include "mpi.h"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/cuSTL/container/HostBuffer.hpp"
#include "pmacc/cuSTL/zone/SphericZone.hpp"
#include <vector>

namespace pmacc
{
    namespace algorithm
    {
        namespace mpi
        {
            /** Reduce algorithm for mpi
             *
             * \tparam dim dimension of the mpi node volume which has to be reduced.
             *
             * This algorithm reduces node-wise. For each node you pass a data container as source
             * and another container of the same size as destination. The result is stored in
             * the destination container of the root node.
             *
             * The data values of the container are reduced independently of each other.
             *
             * The dimension of the container need not be the same as dim.
             *
             */
            template<int dim>
            class Reduce
            {
            private:
                MPI_Comm comm;
                bool m_participate;

            public:
                /** constructor
                 *
                 * \param zone The zone specifies which mpi-nodes participate in the reduce operation.
                 * \param setThisAsRoot Set this node explicitly as root. May only be true for one node.
                 *
                 * if setThisAsRoot is not set mpi chooses the root node.
                 *
                 */
                Reduce(const zone::SphericZone<dim>& zone, bool setThisAsRoot = false);
                ~Reduce();

                /* execute the algorithm
                 *
                 * \param dest destination container
                 * \param src source container
                 * \param ExprOrFunctor functor with two arguments which returns the result of the reduce operation.
                 *
                 * Since only the functor's type is given, the functor must have a standart constructor.
                 *
                 */
                template<typename Type, int conDim, typename ExprOrFunctor>
                void operator()(
                    container::HostBuffer<Type, conDim>& dest,
                    const container::HostBuffer<Type, conDim>& src,
                    ExprOrFunctor) const;

                // Returns whether this node is within the zone.
                inline bool participate() const
                {
                    return m_participate;
                }
                // Returns whether this node is the root node.
                inline bool root() const;
                // Returns the mpi rank of this node.
                inline int rank() const;
            };

        } // namespace mpi
    } // namespace algorithm
} // namespace pmacc

#include "pmacc/cuSTL/algorithm/mpi/Reduce.tpp"
