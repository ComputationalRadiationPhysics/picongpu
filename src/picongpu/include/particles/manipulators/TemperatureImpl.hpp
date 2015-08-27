/**
 * Copyright 2013-2015 Axel Huebl, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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
#include "nvidia/rng/distributions/Normal_float.hpp"
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

template<typename T_ParamClass, typename T_ValueFunctor, typename T_SpeciesType>
struct TemperatureImpl : private T_ValueFunctor
{
    typedef T_ParamClass ParamClass;
    typedef T_SpeciesType SpeciesType;
    typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

    typedef T_ValueFunctor ValueFunctor;

    HINLINE TemperatureImpl(uint32_t currentStep) : isInitialized(false)
    {
        typedef typename SpeciesType::FrameType FrameType;

        GlobalSeed globalSeed;
        mpi::SeedPerRank<simDim> seedPerRank;
        seed = globalSeed() ^
               PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() ^
               TEMPERATURE_SEED;
        seed = seedPerRank(seed) ^ currentStep;

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
                                                                      localCellIdx );
            rng = nvrng::create(rngMethods::Xor(seed, cellIdx), rngDistributions::Normal_float());
            isInitialized = true;
        }
        if (isParticle)
        {
            const float3_X tmpRand = float3_X(rng(),
                                              rng(),
                                              rng());
            const float_X macroWeighting = particle[weighting_];

            const float_X energy = (ParamClass::temperature * UNITCONV_keV_to_Joule) / UNIT_ENERGY;

            // since energy is related to one particle,
            // and our units are normalized for macro particle quanities
            // energy is quite small
            const float_X macroEnergy = macroWeighting * energy;
            // non-rel, MW:
            //    p = m * v
            //            v ~ sqrt(k*T/m), k*T = E
            // => p = sqrt(m)
            //
            // Note on macro particle energies, with weighting w:
            //    p_1 = m_1 * v
            //                v = v_1 = v_w
            //    p_w = p_1 * w
            //    E_w = E_1 * w
            // Since masses, energies and momenta add up linear, we can
            // just take w times the p_1. Take care, E means E_1 !
            // This goes to:
            //    p_w = w * p_1 = w * m_1 * sqrt( E / m_1 )
            //        = sqrt( E * w^2 * m_1 )
            //        = sqrt( E * w * m_w )
            // Which makes sense, since it means that we use a macroMass
            // and a macroEnergy now.
            const float3_X mom = tmpRand * (float_X) math::sqrt(
                                                                precisionCast<sqrt_X > (
                                                                                        macroEnergy *
                                                                                        attribute::getMass(macroWeighting,particle)
                                                                                        )
                                                                );
            ValueFunctor::operator()(particle[momentum_], mom);
        }
    }

private:
    typedef PMacc::nvidia::rng::RNG<rngMethods::Xor, rngDistributions::Normal_float> RngType;
    RngType rng;
    bool isInitialized;
    uint32_t seed;
    DataSpace<simDim> localCells;
};

} //namespace manipulators
} //namespace particles
} //namespace picongpu
