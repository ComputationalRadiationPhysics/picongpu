/**
 * Copyright 2015 Rene Widera
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

#include "simulation_defines.hpp"
#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "nvidia/rng/distributions/Uniform_float.hpp"
#include "mpi/SeedPerRank.hpp"
#include "traits/GetUniqueTypeId.hpp"

namespace picongpu
{
namespace particles
{
namespace manipulators
{

namespace nvrng = nvidia::rng;
namespace rngMethods = nvidia::rng::methods;
namespace rngDistributions = nvidia::rng::distributions;

template< typename T_SpeciesType>
struct RandomPositionImpl
{
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    HINLINE RandomPositionImpl(uint32_t currentStep) : isInitialized(false)
    {
        typedef typename SpeciesType::FrameType FrameType;

        GlobalSeed globalSeed;
        mpi::SeedPerRank<simDim> seedPerRank;
        seed = seedPerRank(globalSeed(), PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid());
        seed ^= POSITION_SEED ^ currentStep;

        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        localCells = subGrid.getLocalDomain().size;
    }

    template<typename T_Particle1, typename T_Particle2>
    DINLINE void operator()(const DataSpace<simDim>& localCellIdx,
                            T_Particle1& particle, T_Particle2&,
                            const bool isParticle, const bool)
    {
        typedef typename T_Particle1::FrameType FrameType;

        if (!isInitialized)
        {
            const uint32_t cellIdx = DataSpaceOperations<simDim>::map(
                                                                      localCells,
                                                                      localCellIdx);
            rng = nvrng::create(rngMethods::Xor(seed, cellIdx), rngDistributions::Uniform_float());
            isInitialized = true;
        }
        if (isParticle)
        {
            floatD_X tmpPos;

            for (uint32_t d = 0; d < simDim; ++d)
                tmpPos[d] = rng();

            particle[position_] = tmpPos;
        }
    }

private:
    typedef PMacc::nvidia::rng::RNG<rngMethods::Xor, rngDistributions::Uniform_float> RngType;
    RngType rng;
    bool isInitialized;
    uint32_t seed;
    DataSpace<simDim> localCells;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
