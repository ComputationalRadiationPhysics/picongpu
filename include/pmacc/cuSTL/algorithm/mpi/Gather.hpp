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
            /**
             */
            template<int dim>
            class Gather
            {
            private:
                MPI_Comm comm;
                std::vector<math::Int<dim>> positions;
                bool m_participate;

                struct CopyToDest
                {
                    template<typename Type, int memDim, class T_Alloc, class T_Copy, class T_Assign>
                    void operator()(
                        const Gather<dim>& gather,
                        container::CartBuffer<Type, memDim, T_Alloc, T_Copy, T_Assign>& dest,
                        std::vector<Type>& tmpDest,
                        int dir,
                        const std::vector<math::Size_t<memDim>>& srcSizes,
                        const std::vector<size_t>& srcOffsets) const;
                };

            public:
                Gather(const zone::SphericZone<dim>& p_zone);
                ~Gather();

                template<
                    typename Type,
                    int memDim,
                    class T_Alloc,
                    class T_Copy,
                    class T_Assign,
                    class T_Alloc2,
                    class T_Copy2,
                    class T_Assign2>
                void operator()(
                    container::CartBuffer<Type, memDim, T_Alloc, T_Copy, T_Assign>& dest,
                    container::CartBuffer<Type, memDim, T_Alloc2, T_Copy2, T_Assign2>& source,
                    int dir = -1) const;

                inline bool participate() const
                {
                    return m_participate;
                }
                inline bool root() const;
                inline int rank() const;
            };

        } // namespace mpi
    } // namespace algorithm
} // namespace pmacc

#include "Gather.tpp"
