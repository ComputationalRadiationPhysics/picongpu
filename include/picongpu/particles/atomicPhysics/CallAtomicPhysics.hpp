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
#include "picongpu/particles/atomicPhysics/RateMatrix.hpp"

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
                // note: renamed to _atomicPhysics temporarily
                _atomicPhysics< > /// atomic physics flag of species from .param file
            >::type
        >;
        using ElectronFrameType = typename ElectronSpecies::FrameType;

        // define entry of atomic data table
        using Items = std::vector< std::pair< uint32_t, float_X > >;
        Items readData( std::string fileName )
        {
            std::ifstream file( fileName );
            if( !file )
            {
                std::cerr << "Atomic physics error: could not open file " << fileName << "\n";
                return Items{};
            }

            Items result;
            float_X stateIndex;
            float_X energyOverGroundState;
            while (file >> stateIndex >> energyOverGroundState)
            {
                uint32_t idx = static_cast< uint32_t >( stateIndex );
                auto item = std::make_pair(
                    idx,
                    energyOverGroundState
                );
                result.push_back( item );
            }
            return result;
        }

        // Constructor loads atomic data
        CallAtomicPhysics()
        {
            // file names
            // hard-coded for now, will be parametrized
            // file name of file containing atomic data
            std::string levelDataFileName = "~/HydrogenLevels.txt";
            std::string transitionDataFileName = "~/HydrogenTransitions.txt";

            // read in atomic data
            // levels
            auto levelDataItems = readData( levelDataFileName );
            // transitions
            auto transitionDataItems = readData( transitionDataFileName );

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

            // remove the last line with state 1
            levelDataItems.pop_back();
            

            // init rate matrix on host and copy to device
            uint32_t firstStateIndex = levelDataItems[0].first;
            rateMatrix = pmacc::memory::makeUnique< RateMatrix >(
                firstStateIndex,
                levelDataItems.size()
            );
            auto rateMatrixHostBox = rateMatrix->getHostDataBox();
            for (uint32_t i = 0; i < levelDataItems.size(); i++ )
            {
                rateMatrixHostBox( levelDataItems[i].first ) = levelDataItems[i].second;
            }
            rateMatrix->syncToDevice();
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
            // TODO: remove
            float_X initialGridWidth = 0.2_X;
            float_X relativeErrorTarget = 0.5_X;

            // hardcoded for now
            // TODO: from param file
            constexpr uint32_t maxNumBins = 2000;

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
                rateMatrix->getDeviceDataBox( ),
                relativeErrorTarget,
                initialGridWidth
            );

            dc.releaseData( ElectronFrameType::getName() );
            dc.releaseData( IonFrameType::getName() );
        }

    private:

        std::unique_ptr< RateMatrix > rateMatrix;

    };

} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
