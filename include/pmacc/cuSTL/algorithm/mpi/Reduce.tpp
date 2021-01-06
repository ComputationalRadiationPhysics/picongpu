/* Copyright 2013-2021 Heiko Burau, Axel Huebl
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

#include "pmacc/cuSTL/container/copier/Memcopy.hpp"

#include "pmacc/mappings/simulation/GridController.hpp"
#include "pmacc/communication/manager_common.hpp"

#include <iostream>
#include <utility>
#include <algorithm>


namespace pmacc
{
    namespace algorithm
    {
        namespace mpi
        {
            template<int dim>
            Reduce<dim>::Reduce(const zone::SphericZone<dim>& p_zone, bool setThisAsRoot) : comm(MPI_COMM_NULL)
            {
                using namespace math;

                auto& con = Environment<dim>::get().GridController();

                typedef std::pair<Int<dim>, bool> PosFlag;
                PosFlag posFlag;
                posFlag.first = (Int<dim>) con.getPosition();
                posFlag.second = setThisAsRoot;

                int numWorldRanks;
                MPI_Comm_size(MPI_COMM_WORLD, &numWorldRanks);
                std::vector<PosFlag> allPositionsFlags(numWorldRanks);

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(
                    (void*) &posFlag,
                    sizeof(PosFlag),
                    MPI_CHAR,
                    (void*) allPositionsFlags.data(),
                    sizeof(PosFlag),
                    MPI_CHAR,
                    MPI_COMM_WORLD));

                std::vector<int> new_ranks;
                int myWorldId;
                MPI_Comm_rank(MPI_COMM_WORLD, &myWorldId);

                this->m_participate = false;
                for(int i = 0; i < (int) allPositionsFlags.size(); i++)
                {
                    Int<dim> pos = allPositionsFlags[i].first;
                    bool flag = allPositionsFlags[i].second;
                    if(!p_zone.within(pos))
                        continue;

                    new_ranks.push_back(i);
                    // if rank i is supposed to be the new root put him at the front
                    if(flag)
                        std::swap(new_ranks.front(), new_ranks.back());
                    if(i == myWorldId)
                        this->m_participate = true;
                }

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();
                if(new_ranks.size())
                {
                    MPI_Group world_group = MPI_GROUP_NULL;
                    MPI_Group new_group = MPI_GROUP_NULL;
                    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
                    MPI_CHECK(MPI_Group_incl(world_group, new_ranks.size(), &(new_ranks.front()), &new_group));
                    MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, new_group, &this->comm));
                    MPI_CHECK(MPI_Group_free(&new_group));
                    MPI_CHECK(MPI_Group_free(&world_group));
                }
            }

            template<int dim>
            Reduce<dim>::~Reduce()
            {
                if(this->comm != MPI_COMM_NULL)
                {
                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    __getTransactionEvent().waitForFinished();
                    MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&this->comm));
                }
            }

            template<int dim>
            bool Reduce<dim>::root() const
            {
                if(!this->m_participate)
                {
                    std::cerr << "error[mpi::Reduce::root()]: this process does not participate in reducing.\n";
                    return false;
                }
                int myId;
                MPI_Comm_rank(this->comm, &myId);
                return myId == 0;
            }

            template<int dim>
            int Reduce<dim>::rank() const
            {
                if(!this->m_participate)
                {
                    std::cerr << "error[mpi::Reduce::rank()]: this process does not participate in reducing.\n";
                    return -1;
                }
                int myId;
                MPI_Comm_rank(this->comm, &myId);
                return myId;
            }

            namespace detail
            {
                template<typename Functor, typename type>
                struct MPI_User_Op
                {
                    static void callback(void* invec, void* inoutvec, int* len, MPI_Datatype*)
                    {
                        Functor functor;
                        type* inoutvec_t = (type*) inoutvec;
                        type* invec_t = (type*) invec;

                        int size = (*len) / sizeof(type);
                        for(int i = 0; i < size; i++)
                        {
                            inoutvec_t[i] = functor(inoutvec_t[i], invec_t[i]);
                        }
                    }
                };

            } // namespace detail

            template<int dim>
            template<typename Type, int conDim, typename Functor>
            void Reduce<dim>::operator()(
                container::HostBuffer<Type, conDim>& dest,
                const container::HostBuffer<Type, conDim>& src,
                Functor) const
            {
                if(!this->m_participate)
                    return;

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                __getTransactionEvent().waitForFinished();

                MPI_Op user_op;
                MPI_CHECK(MPI_Op_create(&detail::MPI_User_Op<Functor, Type>::callback, 1, &user_op));

                MPI_CHECK(MPI_Reduce(
                    &(*src.origin()),
                    &(*dest.origin()),
                    sizeof(Type) * dest.size().productOfComponents(),
                    MPI_CHAR,
                    user_op,
                    0,
                    this->comm));

                MPI_CHECK(MPI_Op_free(&user_op));
            }

        } // namespace mpi
    } // namespace algorithm
} // namespace pmacc
