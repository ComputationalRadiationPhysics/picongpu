/* Copyright 2013-2020 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/fields/background/cellwiseOperation.hpp"
#include "picongpu/fields/FieldJ.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu
{
namespace simulation
{
namespace stage
{

    // Functor to apply the operation for a given ion species
    template< typename T_IonSpecies >
    struct CallCPUStage
    {
        // Define ion species and frame type
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            T_IonSpecies
        >;
        using IonFrameType = typename IonSpecies::FrameType;

        // Define electron species and frame time
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            typename pmacc::particles::traits::ResolveAliasFromSpecies<
                IonSpecies,
                atomicPhysicsElectrons< > /// here will be your flag from .param file
            >::type
        >;
        using ElectronFrameType = typename ElectronSpecies::FrameType;

        // Call the functor
        void operator()( MappingDesc const cellDescription ) const
        {
            std::cout << "Operator(): ion species = " << IonFrameType::getName()
                 << ", electron species = " << ElectronFrameType::getName() << "\n";
            using namespace pmacc;
            DataConnector &dc = Environment<>::get().DataConnector();
            /// NOTE: having false as second parameter will copy to host
            /// (normally is not used as processing is done in kernels on device)
            auto & ions = *dc.get< IonSpecies >( IonFrameType::getName(), false );
            auto & electrons = *dc.get< ElectronSpecies >( ElectronFrameType::getName(), false );
#if( PMACC_CUDA_ENABLED == 1 )
            auto mallocMCBuffer = dc.get< MallocMCBuffer< DeviceHeap > >( MallocMCBuffer< DeviceHeap >::getName(), true );
            auto ionBox = ions.getHostParticlesBox( mallocMCBuffer->getOffset() );
            auto electronBox = electrons.getHostParticlesBox( mallocMCBuffer->getOffset() );
            dc.releaseData( MallocMCBuffer< DeviceHeap >::getName() );
#else
            auto ionBox = ions.getDeviceParticlesBox( );
            auto electronBox = electrons.getDeviceParticlesBox( );
#endif
            process(
                ionBox,
                electronBox,
                cellDescription
            );
            // Copy back to device
            ions.syncToDevice();
            electrons.syncToDevice();
            dc.releaseData( ElectronFrameType::getName() );
            dc.releaseData( IonFrameType::getName() );
        }

        // Process ions and electrons: here the boxes contain all supercells
        template<
            typename T_IonBox,
            typename T_ElectronBox
        >
        void process(
            T_IonBox ionBox,
            T_ElectronBox electronBox,
            MappingDesc const mapper
        ) const
        {
            // For the CPU version, loop over all supercells manually,
            // in a real kernel this will be done in parallel one supercell per block
            auto const guardingSuperCells = mapper.getGuardingSuperCells();
            auto const superCellsCount =
                mapper.getGridSuperCells() - 2 * guardingSuperCells;
            auto const numSuperCells = superCellsCount.productOfComponents();
            for( auto supercellLinearIdx = 0;
                supercellLinearIdx < numSuperCells; ++supercellLinearIdx )
            {
                auto const idxWithGuard = DataSpaceOperations< simDim >::map(
                    superCellsCount,
                    supercellLinearIdx
                ) + guardingSuperCells;
                processSupercell(
                    idxWithGuard,
                    ionBox,
                    electronBox,
                    mapper
                );
            }
        }

        // Process ions and electrons in the supercell with given index
        template<
            typename T_IonBox,
            typename T_ElectronBox
        >
        void processSupercell(
            DataSpace< simDim > const & idx,
            T_IonBox ionBox,
            T_ElectronBox electronBox,
            MappingDesc const mapper
        ) const
        {
            // The real kernel will essentially only have this part,
            // just start with selecting a supercell based on block index

            auto electronFrame = electronBox.getLastFrame( idx );
            // Iterate over ions frames
            auto ionFrame = ionBox.getLastFrame( idx );
            auto ionsInFrame = ionBox.getSuperCell( idx ).getSizeLastFrame();
            while( ionFrame.isValid() )
            {
                // Iterate over ions in a frame, for now sequentially
                // (in the kernel just this loop and index will change)
                for( int ionIdx = 0; ionIdx < ionsInFrame; ionIdx++ )
                {
                    auto ion = ionFrame[ ionIdx ];
                    /// TODO: electron will be selected here, for now use the same one always
                    auto electron = electronFrame[ 0 ];

                    /// Here implement everything using variables ion and electron
                    /// that represent the selected pair
                }

                ionFrame = ionBox.getPreviousFrame( ionFrame );
                ionsInFrame = pmacc::math::CT::volume< SuperCellSize >::type::value;
            }
        }

    };


    //! Test stage for accessing PIC data from a CPU
    class CPUStage
    {
    public:

        CPUStage( MappingDesc const cellDescription ):
            cellDescription( cellDescription )
        {
        }

        void operator( )( uint32_t const step ) const
        {
            using namespace pmacc;
            using SpeciesWithAtomicPhysics = typename pmacc::particles::traits::FilterByFlag<
                VectorAllSpecies,
                atomicPhysicsElectrons< > /// here will be your flag from .param file
            >::type;
            // This will call the CallCPUStage functor for the species from the list
            pmacc::meta::ForEach<
                SpeciesWithAtomicPhysics,
                CallCPUStage< bmpl::_1 >
            > callCPUStage;
            callCPUStage(
                cellDescription
            );

        }

    private:

        //! Mapping for kernels
        MappingDesc cellDescription;

    };

} // namespace stage
} // namespace simulation
} // namespace picongpu
