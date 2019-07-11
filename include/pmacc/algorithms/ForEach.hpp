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

#include "pmacc/traits/GetNumWorkers.hpp"
#include "pmacc/types.hpp"

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
    template<
        uint32_t T_numWorkers,
        uint32_t T_xChunkSize
    >
    struct ForEachFieldValue
    {
        /// TODO
        template<
            typename T_Acc,
            typename T_DataBox,
            typename T_Size
        >
        DINLINE void operator( )(
            T_Acc const & acc,
            T_DataBox box,
            T_Size size
        ) const
        {
            using namespace mappings::threads;

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

            auto const blockIndex{ blockIdx };
            auto blockSize{ T_Size::create( 1 ) };
            blockSize.x( ) = T_xChunkSize;
            constexpr uint32_t numWorkers = T_numWorkers;
            uint32_t const workerIdx = threadIdx.x;

            ForEachIdx<
                IdxConfig<
                T_xChunkSize,
                numWorkers
                >
            >{ workerIdx }(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                    )
            {
                auto virtualWorkerIdx( T_Size::create( 0 ) );
                virtualWorkerIdx.x( ) = linearIdx;
                auto const idx = blockSize * blockIndex + virtualWorkerIdx;
                if( idx.x() < size.x() )
                    accFunctor(
                        acc,
                        box( idx )
                    );
            }
            );



            //using SuperCellSize = typename T_Mapping::SuperCellSize;
            //constexpr uint32_t dim = SuperCellSize::dim;
            //using namespace mappings::threads;
            //constexpr uint32_t cellsPerSuperCell = pmacc::math::CT::volume< SuperCellSize >::type::value;
            //uint32_t const workerIdx = threadIdx.x;
            //DataSpace< dim > const superCellIdx =
            //    mapper.getSuperCellIndex( DataSpace< dim >( blockIdx ) );
            //DataSpace< dim > const blockCell = superCellIdx * SuperCellSize::toRT( );

            //DataSpace< dim > const localSuperCellOffset =
            //    superCellIdx - mapper.getGuardingSuperCells( );
            //auto accFunctor = functor(
            //    acc,
            //    localSuperCellOffset,
            //    WorkerCfg< T_numWorkers >{ workerIdx }
            //);

            //ForEachIdx<
            //    IdxConfig<
            //        cellsPerSuperCell,
            //    T_numWorkers
            //    >
            //>{ workerIdx }(
            //    [&](
            //        uint32_t const linearIdx,
            //        uint32_t const
            //    )
            //    {
            //        auto const cellIdx = DataSpaceOperations< SuperCellSize::dim >::template map< SuperCellSize >( linearIdx );
            //        accFunctor(
            //            acc,
            //            field( blockCell + cellIdx )
            //        );
            //    }
            //);
        }
    };

} //namespace detail
} //namespace acc

    template<
        typename T_Buffer,
        typename T_Functor
    >
    void forEachValue(
        T_Buffer && buffer,
        T_Functor && functor,
        cudaStream_t stream = 0
    )
    {
        auto const linearSize = buffer.getCurrentSize( );
        auto const size = buffer.getCurrentDataSpace( linearSize );
        if( size.productOfComponents() == 0 )
            return;

        auto gridSize = size;
        constexpr auto xChunkSize = 256u;
        constexpr auto numWorkers = traits::GetNumWorkers< xChunkSize >::value;
        gridSize.x( ) = ( gridSize.x( ) + xChunkSize - 1 ) / xChunkSize;
        auto destBox = this->destination->getDataBox( );
        CUPLA_KERNEL(
            ForEachFieldValue<
                numWorkers,
                xChunkSize
            >
        )(
            gridSize,
            numWorkers,
            0,
            stream
        )(
            buffer.getDataBox( ),
            functor
        );

        //using MappingDesc = decltype( box.getCellDescription( ) );
        //AreaMapping<
        //    CORE + BORDER + GUARD,
        //    MappingDesc
        //> mapper( field.getCellDescription() );

        //using SuperCellSize = typename MappingDesc::SuperCellSize;

        //constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
        //    pmacc::math::CT::volume< SuperCellSize >::type::value
        //>::value;

        //PMACC_KERNEL( acc::detail::ForEachFieldValue< numWorkers >{ } )(
        //    mapper.getGridDim(),
        //    numWorkers
        //)(
        //    functor,
        //    mapper,
        //    box.getDeviceDataBox( )
        //);
    }


} // namespace algorithm
} // namespace fields
} // namespace pmacc
