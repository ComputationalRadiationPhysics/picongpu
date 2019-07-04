/* Copyright 2019 Sergei Bastrakov
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

#include "pmacc/Environment.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"

#include <boost/mpl/placeholders.hpp>

#include <cstdint>


namespace pmacc
{
namespace fields
{
namespace algorithm
{
namespace acc
{
namespace detail
{

    /// TODO
    template< uint32_t T_numWorkers >
    struct ForEachFieldValue
    {
        /// TODO
        template<
            typename T_Acc,
            typename T_Functor,
            typename T_Mapping,
            typename T_FieldBox
        >
        DINLINE void operator( )(
            T_Acc const & acc,
            T_Functor functor,
            T_Mapping const mapper,
            T_FieldBox field
        ) const
        {
            using SuperCellSize = typename T_Mapping::SuperCellSize;
            constexpr uint32_t dim = SuperCellSize::dim;
            using namespace mappings::threads;
            constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume< SuperCellSize >::type::value;
            uint32_t const workerIdx = threadIdx.x;
            DataSpace< dim > const superCellIdx =
                mapper.getSuperCellIndex( DataSpace< dim >( blockIdx ) );
            DataSpace< dim > const blockCell = superCellIdx * SuperCellSize::toRT( );

            DataSpace< dim > const localSuperCellOffset =
                superCellIdx - mapper.getGuardingSuperCells( );
            auto accFunctor = functor(
                acc,
                localSuperCellOffset,
                WorkerCfg< T_numWorkers >{ workerIdx }
            );

            ForEachIdx<
                IdxConfig<
                    cellsPerSuperCell,
                T_numWorkers
                >
            >{ workerIdx }(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    auto const cellIdx = DataSpaceOperations< SuperCellSize::dim >::template map< SuperCellSize >( linearIdx );
                    accFunctor(
                        acc,
                        field( blockCell + cellIdx )
                    );
                }
            );
        }
    };

} //namespace detail
} //namespace acc

    template<
        typename T_Field,
        typename T_Functor
    >
    void forEach(
        T_Field && field,
        T_Functor && functor
    )
    {
        using MappingDesc = decltype( field.getCellDescription( ) );
        AreaMapping<
            CORE + BORDER + GUARD,
            MappingDesc
        > mapper( field.getCellDescription() );

        using SuperCellSize = typename MappingDesc::SuperCellSize;

        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume< SuperCellSize >::type::value
        >::value;

        PMACC_KERNEL( acc::detail::ForEachFieldValue< numWorkers >{ } )(
            mapper.getGridDim(),
            numWorkers
        )(
            functor,
            mapper,
            field.getDeviceDataBox( )
        );
    }


} // namespace algorithm
} // namespace fields
} // namespace pmacc
