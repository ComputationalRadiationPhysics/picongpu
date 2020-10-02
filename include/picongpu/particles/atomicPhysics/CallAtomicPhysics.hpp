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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/atomicPhysics/AtomicPhysics.kernel"
#include "picongpu/particles/atomicPhysics/AtomicData.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{

    // Functor to apply the operation for a given ion species
    // @tparam T_ConfigNumberDataType ... data type used for index storage by ConfigNumber object of ion species
    template<
        typename T_IonSpecies
        >
    struct CallAtomicPhysics
    {
        // Define ion species and frame type datatype for later access
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            T_IonSpecies
        >;
        using IonFrameType = typename IonSpecies::FrameType;
        // TODO: get this from species
        using T_ConfigNumberDataType = uint64_t;
        static constexpr uint8_t T_atomicNumber = 6u;

        // Define electron species and frame type datatype for later access
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            typename pmacc::particles::traits::ResolveAliasFromSpecies<
                IonSpecies,
                // note: renamed to _atomicPhysics temporarily
                _atomicPhysics< > /// atomic physics flag of species from .param file
            >::type
        >;
        using ElectronFrameType = typename ElectronSpecies::FrameType;

        // define entry of atomic data table
        using States = typename std::vector< std::pair< uint64_t, float_X > >;
        using Transitions = typename std::vector< std::tuple<
            uint64_t,
            uint64_t,
            float_X,
            float_X,
            float_X,
            float_X,
            float_X,
            float_X
            > >;

    private:
        std::unique_ptr< AtomicData<
            T_atomicNumber,
            T_ConfigNumberDataType
        > > atomicData;

    public:
        // read Data method: atomic states, BEWARE: does not convert to internal units
        States readStateData( std::string fileName )
        {
            std::ifstream file( fileName );
            if( !file )
            {
                std::cerr << "Atomic physics error: could not open file " << fileName << "\n";
                return States{};
            }

            States result;
            double stateIndex;      // TODO: catch overflow if full uint64 is used
            float_X energyOverGroundState;

            while ( file >> stateIndex >> energyOverGroundState )
            {
                uint64_t idx = static_cast< uint64_t >( stateIndex );
                auto item = std::make_pair(
                    idx,
                    energyOverGroundState
                );
                result.push_back( item );
            }
            return result;
        }

        // read Data method: atomic transitions, BEWARE: does not convert to internal units
        Transitions readTransitionData( std::string fileName )
        {
            std::ifstream file( fileName );
            if( !file )
            {
                std::cerr << "Atomic physics error: could not open file " << fileName << "\n";
                return Transitions{};
            }

            double idxLower;
            double idxUpper;
            float_X collisionalOscillatorStrength;

            // gauntCoeficients
            float_X cinx1;
            float_X cinx2;
            float_X cinx3; 
            float_X cinx4;
            float_X cinx5;

            Transitions result;

            while ( file >> idxLower >> idxUpper
                >> collisionalOscillatorStrength
                >> cinx1 >> cinx2 >> cinx3 >> cinx4 >> cinx5
                )
            {
                uint64_t stateIndexLower = static_cast< uint64_t >( idxLower );
                uint64_t stateIndexUpper = static_cast< uint64_t >( idxUpper );

                auto item = std::make_tuple(
                    stateIndexLower,
                    stateIndexUpper,
                    collisionalOscillatorStrength,
                    cinx1,
                    cinx2,
                    cinx3,
                    cinx4,
                    cinx5
                );
                result.push_back( item );
            }
            return result;
        }

        // Constructor loads atomic data
        CallAtomicPhysics()
        {
            /* file names of file containing atomic data
             *
             * hard-coded for now, get out of param files later
             */
            std::string levelDataFileName = "/home/marre55/CarbonLevels.txt";
            std::string transitionDataFileName = "/home/marre55/CarbonTransitions.txt";

            // read in atomic data
            // levels
            auto levelDataItems = readStateData( levelDataFileName );
            // transitions
            auto transitionDataItems = readTransitionData( transitionDataFileName );

            // check whether read was sucessfull
            if( levelDataItems.empty() )
            {
                std::cout << "Could not read the atomic level data. Check given filename.\n";
                return;
            }
            if ( transitionDataItems.empty() )
            {
                std::cout << "Could not read the atomic transition data. Check given filename.\n";
                return;
            }

            uint32_t const maxNumberStates = levelDataItems.size();
            uint32_t const maxNumberTransitions = transitionDataItems.size();

            // init rate matrix on host and copy to device

            // create atomic Data storage class on host
            atomicData = pmacc::memory::makeUnique<
                AtomicData<
                    T_atomicNumber,
                    T_ConfigNumberDataType
                >
            >(
                levelDataItems.size(),
                transitionDataItems.size()
            );

            // get acess to data box on host side
            // init is empty
            auto atomicDataHostBox = atomicData->getHostDataBox<
                maxNumberStates,
                maxNumberTransitions
            >( 0u, 0u );

            // fill atomic data into dataBox
            for ( uint32_t i = 0; i < levelDataItems.size(); i++ )
            {
                atomicDataHostBox.addLevel(
                    levelDataItems[i].first,
                    levelDataItems[i].second
                    );
            }

            // fill atomic transition data into dataBox
            for ( uint32_t i = 0; i < transitionDataItems.size(); i++ )
            {
                atomicDataHostBox.addTransition(
                    std::get< 0 >( transitionDataItems[i] ),
                    std::get< 1 >( transitionDataItems[i] ),
                    std::get< 2 >( transitionDataItems[i] ),
                    std::get< 3 >( transitionDataItems[i] ),
                    std::get< 4 >( transitionDataItems[i] ),
                    std::get< 5 >( transitionDataItems[i] ),
                    std::get< 6 >( transitionDataItems[i] ),
                    std::get< 7 >( transitionDataItems[i] )
                    );
            }

            // copy data to device buffer of atomicData
            atomicData->syncToDevice();
        }

        // Call functor, will be called in MySimulation once per time step
        void operator()(
            uint32_t const step,
            MappingDesc const cellDescription
        ) const
        {
            using namespace pmacc;

            DataConnector & dc = Environment<>::get().DataConnector();

            auto & ions = *dc.get< IonSpecies >(
                IonFrameType::getName(),
                true
            );
            auto & electrons = *dc.get< ElectronSpecies >(
                ElectronFrameType::getName(),
                true
            );

            AreaMapping<
                CORE + BORDER, // full local domain, no guards
                MappingDesc
            > mapper( cellDescription );
            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< MappingDesc::SuperCellSize >::type::value
            >::value;

            // hardcoded for now
            // TODO: make available as options from param file
            constexpr float_X initialGridWidth = 0.2_X; // unit: ATOMIC_UNIT_ENERGY
            constexpr float_X relativeErrorTarget = 0.5_X; // unit: 1/s /( 1/( m^3 * ATOMIC_UNIT_ENERGY ) )
            constexpr uint16_t maxNumBins = 2000;

            using Kernel = AtomicPhysicsKernel<
                numWorkers,
                maxNumBins
            >;
            auto kernel = Kernel{
                RngFactoryInt{ step },
                RngFactoryFloat{ step }
            };

            PMACC_KERNEL( kernel )(
                mapper.getGridDim(), // how many blocks = how many supercells in local domain
                numWorkers           // how many threads per block
            )(
                electrons.getDeviceParticlesBox( ),
                ions.getDeviceParticlesBox( ),
                mapper,
                atomicData->getDeviceDataBox( ),
                initialGridWidth, // unit: J, SI
                relativeErrorTarget // unit: 1/s /( 1/( m^3 * J ) ), SI
            );

            dc.releaseData( ElectronFrameType::getName() );
            dc.releaseData( IonFrameType::getName() );
        }

    };

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
