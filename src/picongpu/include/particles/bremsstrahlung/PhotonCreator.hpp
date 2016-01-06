/**
 * Copyright 2015 Heiko Burau, Marco Garten
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

#include "Cached_F2P_Interp.hpp"
#include "fields/numericalCellTypes/GetNumericalFieldPos.hpp"
#include "particles/bremsstrahlung/SynchrotronFunctions.hpp"
#include "algorithms/math/defines/sqrt.hpp"
#include "algorithms/math/defines/dot.hpp"
#include "algorithms/math/defines/cross.hpp"
#include "traits/frame/GetMass.hpp"
#include "traits/frame/GetCharge.hpp"
#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "cuSTL/container/DeviceBuffer.hpp"

// Random number generator
#include "particles/ionization/ionizationMethods.hpp"

namespace picongpu
{
namespace particles
{
namespace bremsstrahlung
{

/**
 *
 * \tparam T_ElectronSpecies
 * \tparam T_PhotonSpecies
 */
template<typename T_ElectronSpecies, typename T_PhotonSpecies>
struct PhotonCreator
{
    typedef T_ElectronSpecies ElectronSpecies;
    typedef T_PhotonSpecies PhotonSpecies;

    typedef typename ElectronSpecies::FrameType FrameType;
    typedef typename MappingDesc::SuperCellSize SuperCellSize;

    /* specify field to particle interpolation scheme */
	typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType,interpolation<> >::type
        >::type Field2ParticleInterpolation;

    typedef FieldE::ValueType ValueType_E;
	typedef FieldB::ValueType ValueType_B;

    typedef Cached_F2P_Interp<
        container::DeviceBuffer<ValueType_E, simDim>,
        Field2ParticleInterpolation,
        GetNumericalFieldPos<fieldSolver::NumericalCellType, FIELD_TYPE_E>,
        0> CachedFieldE;
    typedef Cached_F2P_Interp<
        container::DeviceBuffer<ValueType_B, simDim>,
        Field2ParticleInterpolation,
        GetNumericalFieldPos<fieldSolver::NumericalCellType, FIELD_TYPE_B>,
        1> CachedFieldB;

private:
    PMACC_ALIGN(cachedFieldE, CachedFieldE);
    PMACC_ALIGN(cachedFieldB, CachedFieldB);
    PMACC_ALIGN(curF_1, SynchrotronFunctions::SyncFuncCursor);
    PMACC_ALIGN(curF_2, SynchrotronFunctions::SyncFuncCursor);

    /* random number generator for Monte Carlo */
    typedef ionization::RandomNrForMonteCarlo<T_ElectronSpecies> RandomGen;
    /* \todo fix: cannot PMACC_ALIGN() because it seems to be too large */
    RandomGen randomGen;

    float3_X photon_mom;

    DINLINE float_X emission_prob(
        const float_X delta,
        const float_X chi,
        const float_X gamma,
        const float_X mass,
        const float_X charge) const
    {
        // quantum
        //const float_X z = float_X(2.0/3.0) * delta / ((float_X(1.0) - delta) * chi);
        // classical
        const float_X z = float_X(2.0/3.0) * delta / chi;

        // quantum
        /*return DELTA_T * charge*charge * mass*SPEED_OF_LIGHT / (float_X(4.0)*PI*EPS0 * HBAR*HBAR) *
            float_X(1.732050807) / (float_X(2.0) * PI) * chi/gamma * (float_X(1.0) - delta) / delta *
            (this->curF_1[z] + float_X(1.5) * delta * chi * z * this->curF_2[z]);*/
        // classical
        return DELTA_T * charge*charge * mass*SPEED_OF_LIGHT / (float_X(4.0)*PI*EPS0 * HBAR*HBAR) *
            float_X(1.732050807) / (float_X(2.0) * PI) * chi/gamma / delta * this->curF_1[z];
    }

    DINLINE float_X emission_prob_modified(
        const float_X z,
        const float_X chi,
        const float_X gamma,
        const float_X mass,
        const float_X charge) const
    {
        return float_X(3.0) * z*z * emission_prob(
            z*z*z,
            chi, gamma, mass, charge);
    }

public:
    /* host constructor initializing member : random number generator */
    PhotonCreator(
        const container::DeviceBuffer<ValueType_E, simDim>& fieldE,
        const container::DeviceBuffer<ValueType_B, simDim>& fieldB,
        const SynchrotronFunctions::SyncFuncCursor& curF_1,
        const SynchrotronFunctions::SyncFuncCursor& curF_2,
        const uint32_t currentStep)
            : cachedFieldE(fieldE), cachedFieldB(fieldB), curF_1(curF_1), curF_2(curF_2), randomGen(currentStep)
    {}

    /** Initialization function on device
     *
     * \brief Cache EM-fields on device
     *         and initialize possible prerequisites for ionization, like e.g. random number generator.
     *
     * This function will be called inline on the device which must happen BEFORE threads diverge
     * during loop execution. The reason for this is the `__syncthreads()` call which is necessary after
     * initializing the E-/B-field shared boxes in shared memory.
     */
    DINLINE void init(const PMacc::math::Int<simDim>& blockCell, const int& linearThreadIdx, const PMacc::math::Int<simDim>& cellOffset)
    {
        this->cachedFieldE.init(blockCell, linearThreadIdx);
        this->cachedFieldB.init(blockCell, linearThreadIdx);
        /* initialize random number generator with the local cell index in the simulation*/
        this->randomGen.init(cellOffset);
    }

    /** Return the number of target particles to be created from each source particle.
     *
     * Called for each frame of the source species.
     *
     * @param sourceFrame Frame of the source species
     * @param localIdx Index of the source particle within frame
     * @return number of particle to be created from each source particle
     */
    DINLINE unsigned int numNewParticles(FrameType& sourceFrame, int localIdx)
    {
        using namespace PMacc::algorithms;

        PMACC_AUTO(particle, sourceFrame[localIdx]);

        const PMacc::math::Int<simDim> localCell =
            PMacc::math::MapToPos<simDim>()(SuperCellSize(), particle[localCellIdx_]);

        /* get fields at the particles's position */
        const ValueType_E fieldE = this->cachedFieldE(particle[position_], localCell);
        const ValueType_B fieldB = this->cachedFieldB(particle[position_], localCell);

        const float3_X mom = particle[momentum_] / particle[weighting_];
        const float_X mom2 = math::dot(mom, mom);
        const float3_X mom_norm = mom * math::rsqrt(mom2);
        const float_X mass = frame::getMass<FrameType>();
        const float_X mc = mass * SPEED_OF_LIGHT;

        const float_X gamma = math::sqrt(float_X(1.0) + mom2 / (mc*mc));
        const float3_X vel = mom / (gamma * mass); // low accuracy?

        const float3_X lorentz = fieldE + math::cross(vel, fieldB);
        const float_X lorentz2 = math::dot(lorentz, lorentz);
        const float_X fieldE_long = math::dot(mom_norm, fieldE);

        const float_X H_eff = math::sqrt(lorentz2 - fieldE_long*fieldE_long);

        const float_X charge = math::abs(frame::getCharge<FrameType>());
        const float_X c = SPEED_OF_LIGHT;
        // Schwinger limit, unit: V/m
        const float_X E_S = mass*mass * c*c*c / (charge * HBAR);

        const float_X chi = gamma * H_eff / E_S;

        const float_X z = this->randomGen();
        if(z == float_X(0.0) || z == float_X(1.0))
            return 0;

        const float_X x = emission_prob_modified(z, chi, gamma, mass, charge);

        if(this->randomGen() < x)
        {
            //printf("E_S: %f, H_eff: %f, chi: %f, gamma: %f, delta: %f, p: %f\n", E_S, H_eff, chi, gamma, z*z*z, x);

            this->photon_mom = mom_norm * z*z*z * mass*c * gamma * particle[weighting_];
            return 1;
        }

        return 0;
    }

    /** Functor implementation.
     *
     * Called once for each single particle creation.
     *
     * \tparam Electron type of electron which creates the photon
     * \tparam Photon type of photon that is created
     */
    template<typename Electron, typename Photon>
    DINLINE void operator()(Electron& electron, Photon& photon) const
    {
        photon[multiMask_] = 1;
        photon[localCellIdx_] = electron[localCellIdx_];
        photon[position_] = electron[position_];
        photon[momentum_] = this->photon_mom;
        electron[momentum_] -= this->photon_mom;
        photon[weighting_] = electron[weighting_];
    }
};

} // namespace bremsstrahlung
} // namespace particles
} // namespace picongpu
