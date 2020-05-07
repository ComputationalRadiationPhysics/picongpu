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

#include <picongpu/param/grid.param>

#include <picongpu/fields/background/cellwiseOperation.hpp>
#include <picongpu/fields/FieldJ.hpp>

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/type/Area.hpp>

// modules necessary for random number generators from pmacc
#include <pmacc/random/methods/methods.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/RNGProvider.hpp>

#include <cstdint>

namespace picongpu
{
namespace traits
{
    /** specialization of the UsesRNG trait
     * --> atomicPhysics module uses random number generation
     */
    template<>
    struct UsesRNG<simulation::stage::CallAtomicPhysics> : public boost::true_type
    {
    };
} // namespace traits

namespace simulation
{
namespace stage
{

    // Functor to apply the operation for a given ion species
    template< typename T_IonSpecies >
    struct CallAtomicPhysics
    {
        // Define ion species and frame type datatype for later access
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            T_IonSpecies
        >;
        using IonFrameType = typename IonSpecies::FrameType;

        // Define electron species and frame type datatype for later access
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            typename pmacc::particles::traits::ResolveAliasFromSpecies<
                IonSpecies,
                atomicPhysics< > /// here will be your flag from .param file
            >::type
        >;
        using ElectronFrameType = typename ElectronSpecies::FrameType;

        // get specialisation of ConfigNumber class used in this species
        using IonSpeciesAtomicConfigNumber =
            pmacc::particles::traits::ResolveAliasFromSpecies<
                IonSpecies,
                atomicConfigNumber< >
                /* atomicConfigNumber is alias(interface name) for specific
                specialisation of ConfigNumber of this specific species*/
            >;
        /* get T_DataType used as parameter in ConfigNumber.hpp via public
        typedef in class */
        using ConfigNumberDataType =
            typename IonSpeciesAtomicConfigNumber::DataType;

        // random number generator(RNG) Factory as defined in random.param
        using RNGFactory = pmacc::random::RNGProvider<
            simDim,
            random::Generator
            >;
        using Distribution = pmacc::random::distributions::Uniform<
            ConfigNumberDataType
            >;
        // type of random number Generator extracted from RNGFactory
        using RandomGen = typename RNGFactory::GetRandomType<
            Distribution
            >::type;
        // actual random number Generator defined as attribute and initialised
        RandomGen randomGen = RNGFactory::createRandom< Distribution >();

        // Call functor, will be called in MySimulation once per time step
        void operator()( MappingDesc const cellDescription ) const
        {
            // organisation of particle data retrival

            // debug console output
            std::cout << "Operator(): ion species = " << IonFrameType::getName()
                 << ", electron species = " << ElectronFrameType::getName() << "\n";

            using namespace pmacc;

            DataConnector &dc = Environment<>::get().DataConnector();

            /// NOTE: having false as second parameter will copy to host
            /// (normally is not used as processing is done in kernels on device)
            auto & ions = *dc.get< IonSpecies >( IonFrameType::getName(), false );
            auto & electrons = *dc.get< ElectronSpecies >( ElectronFrameType::getName(), false );

            // depending on whether using gpus(CUDA) or cpu(no cuda) different data retrival
#if( PMACC_CUDA_ENABLED == 1 )
            auto mallocMCBuffer = dc.get< MallocMCBuffer< DeviceHeap > >( MallocMCBuffer< DeviceHeap >::getName(), true );
            auto ionBox = ions.getHostParticlesBox( mallocMCBuffer->getOffset() );
            auto electronBox = electrons.getHostParticlesBox( mallocMCBuffer->getOffset() );
            dc.releaseData( MallocMCBuffer< DeviceHeap >::getName() );
#else
            auto ionBox = ions.getDeviceParticlesBox( );
            auto electronBox = electrons.getDeviceParticlesBox( );
#endif

            // actual call of algorithm process, future kernel call
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

                    double timeRemaining;
                    timeRemaining = static_cast< double >(
                        picongpu::SI::DELTA_T_SI
                    );

                    auto configNumber = ion[atomicConfigNumber_];
                    auto numberPossibilities = configNumber.numberStates();
                    randomGen(  )
                }

                ionFrame = ionBox.getPreviousFrame( ionFrame );
                ionsInFrame = pmacc::math::CT::volume< SuperCellSize >::type::value;
            }
        }

    };


    //! Test stage for accessing PIC data from a CPU
    class AtomicPhysics
    {
    public:

        AtomicPhysics( MappingDesc const cellDescription ):
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
            // This will call the AtomicPhysics functor for the species from the list
            pmacc::meta::ForEach<
                SpeciesWithAtomicPhysics,
                CallAtomicPhysics< bmpl::_1 >
            > callAtomicPhysics;
            callAtomicPhysics(
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
