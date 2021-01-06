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

#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/dimensions/SuperCellDescription.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** execute a functor for each cell of a domain
     *
     * the user functor is executed on each elements of the full domain (GUARD +CORE)
     *
     * @tparam T_DataDomain pmacc::SuperCellDescription, compile time data domain
     *                      description with a CORE and GUARD
     * @tparam T_numWorkers number of workers
     */
    template<typename T_DataDomain, uint32_t T_numWorkers>
    class ThreadCollective
    {
    private:
        // size of the CORE (in elements per dimension)
        using CoreDomainSize = typename T_DataDomain::SuperCellSize;
        // full size of the domain including the GUARD (in elements per dimension)
        using DomainSize = typename T_DataDomain::FullSuperCellSize;
        // offset (in elements per dimension) from the GUARD origin to the CORE
        using OffsetOrigin = typename T_DataDomain::OffsetOrigin;

        static constexpr uint32_t numWorkers = T_numWorkers;
        static constexpr uint32_t dim = T_DataDomain::Dim;

        PMACC_ALIGN(m_workerIdx, const uint32_t);

    public:
        /** constructor
         *
         * @param workerIdx index of the worker
         */
        DINLINE ThreadCollective(uint32_t const workerIdx) : m_workerIdx(workerIdx)
        {
        }

        /** execute the user functor for each element in the full domain
         *
         * @tparam T_Functor type of the user functor, must have a `void operator()`
         *                   with as many arguments as args contains
         * @tparam T_Args type of the arguments, each type must implement an operator
         *                 `template<typename T, typename R> R operator(T)`
         *
         * @param functor user defined functor
         * @param args arguments passed to the functor
         *             The method `template<typename T, typename R> R operator(T)`
         *             is called for each argument, the result is passed to the
         *             functor `functor::operator()`.
         *             `T` is a N-dimensional vector of an index relative to the origin
         *             of data domain GUARD
         */
        template<typename T_Functor, typename... T_Args, typename T_Acc>
        DINLINE void operator()(T_Acc const& acc, T_Functor& functor, T_Args&&... args)
        {
            using namespace mappings::threads;
            ForEachIdx<IdxConfig<math::CT::volume<DomainSize>::type::value, numWorkers>>{m_workerIdx}(
                [&](uint32_t const linearIdx, uint32_t const) {
                    /* offset (in elements) of the current processed element relative
                     * to the origin of the core domain
                     */
                    DataSpace<dim> const offset(
                        DataSpaceOperations<dim>::template map<DomainSize>(linearIdx) - OffsetOrigin::toRT());
                    functor(acc, args(offset)...);
                });
        }
    };

} // namespace pmacc
