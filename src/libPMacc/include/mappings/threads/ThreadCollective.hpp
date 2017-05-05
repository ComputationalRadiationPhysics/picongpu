/* Copyright 2013-2017 Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
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


#pragma once

#include "mappings/threads/ForEachIdx.hpp"
#include "mappings/threads/IdxConfig.hpp"
#include "dimensions/SuperCellDescription.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "dimensions/DataSpace.hpp"
#include "pmacc_types.hpp"

namespace PMacc
{

/** execute a functor on a domain
 *
 * For each argument passed to the functor the `operator()` is called.
 * This is a **collective** functor and needs to be called by all worker within a block.
 *
 * @tparam T_BlockDomain domain description, type mist contain the definitions `SuperCellSize`,
 *                       `FullSuperCellSize`, `OffsetOrigin` and `Dim`
 * @tparam T_numWorkers number of workers which are used to execute this functor
 */
template<
    typename T_BlockDomain,
    uint32_t T_numWorkers = math::CT::volume<typename T_BlockDomain::SuperCellSize>::type::value
>
class ThreadCollective
{
private:
    /** size of the inner domain */
    using SuperCellSize = typename T_BlockDomain::SuperCellSize;

    /** full size of the domain including the guards around SuperCellSize */
    using FullSuperCellSize = typename T_BlockDomain::FullSuperCellSize;

    /** size of the negative guard */
    using OffsetOrigin = typename T_BlockDomain::OffsetOrigin;

    /** number of worker performing the functior */
    static constexpr uint32_t numWorkers = T_numWorkers;

    /** dimensionality of the index domain */
    static constexpr uint32_t dim = T_BlockDomain::Dim;

    /** index of the worker: range [0;numWorkers) */
    PMACC_ALIGN( m_workerIdx, uint32_t const );

public:

    /** constructor
     *
     * @param workerIdx worker index
     */
    DINLINE ThreadCollective( uint32_t const workerIdx ) :
        m_workerIdx( workerIdx )
    { }

    /** constructor
     *
     * @warning this constructor is **deprecated** (assuming that the index domain
     *          and the number of workers needs to be euqal is not good and avoid the
     *          usage of alpaka)
     *
     * @param nDimWorkerIdx n dimensional worker index
     */
    DINLINE ThreadCollective( DataSpace< dim > const nDimWorkerIdx ) :
        m_workerIdx( DataSpaceOperations< dim >::template map< SuperCellSize >( nDimWorkerIdx ) )
    { }

    /** execute a functor for each data point
     *
     * @tparam T_Functor type of the functor
     * @tparam T_Args type of arguments given to the functor
     *
     * @param functor functor which is called for each data element in the domain
     *                spanned by T_BlockDomain
     * @param args the result of each argument of the `operator( relativeNDimensionalIdx )` call
     *             with the relative index inside `T_BlockDomain` based on
     *             the origin `T_BlockDomain::OffsetOrigin` is passed to the functor
     */
    template<
        typename T_Functor,
        typename ... T_Args
    >
    DINLINE void operator()(
        T_Functor & functor,
        T_Args && ... args
    )
    {
        mappings::threads::ForEachIdx<
            mappings::threads::IdxConfig<
                math::CT::volume<FullSuperCellSize>::type::value,
                numWorkers
            >
        >{ m_workerIdx }(
            [&]( uint32_t const linearIdx, uint32_t const )
            {
                DataSpace< dim > const relativeIdx(
                    DataSpaceOperations< dim >::template map< FullSuperCellSize >( linearIdx ) -
                    OffsetOrigin::toRT( )
                );
                functor( args( relativeIdx ) ... );
            }
        );
    }
};

}//namespace PMacc
