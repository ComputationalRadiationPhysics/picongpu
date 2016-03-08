/**
 * Copyright 2015-2016 Heiko Burau
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
#include "SynchrotronFunctions.hpp"
#include "pmacc_types.hpp"
#include "algorithms/Gamma.hpp"
#include "algorithms/math/defines/sqrt.hpp"
#include "algorithms/math/defines/dot.hpp"
#include "algorithms/math/defines/cross.hpp"
#include "traits/frame/GetMass.hpp"
#include "traits/frame/GetCharge.hpp"

#include "random/methods/XorMin.hpp"
#include "random/distributions/Uniform.hpp"
#include "random/RNGProvider.hpp"

#include "traits/Resolve.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "compileTime/conversion/TypeToPointerPair.hpp"
#include "memory/boxes/DataBox.hpp"

namespace picongpu
{
namespace particles
{
namespace synchrotronPhotons
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

    /* specify field to particle interpolation scheme */
    typedef typename PMacc::traits::Resolve<
        typename GetFlagType<FrameType,interpolation<> >::type
        >::type Field2ParticleInterpolation;

    /* margins around the supercell for the interpolation of the field on the cells */
    typedef typename GetMargin<Field2ParticleInterpolation>::LowerMargin LowerMargin;
    typedef typename GetMargin<Field2ParticleInterpolation>::UpperMargin UpperMargin;

    /* relevant area of a block */
    typedef SuperCellDescription<
        typename MappingDesc::SuperCellSize,
        LowerMargin,
        UpperMargin
        > BlockArea;

    BlockArea BlockDescription;

    typedef MappingDesc::SuperCellSize TVec;

    typedef FieldE::ValueType ValueType_E;
    typedef FieldB::ValueType ValueType_B;

private:
    /* global memory EM-field device databoxes */
    PMACC_ALIGN(eBox, FieldE::DataBoxType);
    PMACC_ALIGN(bBox, FieldB::DataBoxType);
    /* shared memory EM-field device databoxes */
    PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize,1> >);
    PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize,0> >);

    PMACC_ALIGN(curF_1, SynchrotronFunctions::SyncFuncCursor);
    PMACC_ALIGN(curF_2, SynchrotronFunctions::SyncFuncCursor);

    PMACC_ALIGN(photon_mom, float3_X);

    /* random number generator */
    typedef PMacc::random::RNGProvider<simDim, PMacc::random::methods::XorMin> RNGFactory;
    typedef PMacc::random::distributions::Uniform<float> Distribution;
    typedef typename RNGFactory::GetRandomType<Distribution>::type RandomGen;
    RandomGen randomGen;

public:
    /* host constructor initializing member : random number generator */
    PhotonCreator(
        const SynchrotronFunctions::SyncFuncCursor& curF_1,
        const SynchrotronFunctions::SyncFuncCursor& curF_2)
            : curF_1(curF_1),
              curF_2(curF_2),
              photon_mom(float3_X::create(0)),
              randomGen(RNGFactory::createRandom<Distribution>())
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        /* initialize pointers on host-side E-(B-)field databoxes */
        FieldE* fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));
        FieldB* fieldB = &(dc.getData<FieldB > (FieldB::getName(), true));
        /* initialize device-side E-(B-)field databoxes */
        eBox = fieldE->getDeviceDataBox();
        bBox = fieldB->getDeviceDataBox();
    }

    /** Initialization function on device
     *
     * \brief Cache EM-fields on device
     *         and initialize possible prerequisites for ionization, like e.g. random number generator.
     *
     * This function will be called inline on the device which must happen BEFORE threads diverge
     * during loop execution. The reason for this is the `__syncthreads()` call which is necessary after
     * initializing the E-/B-field shared boxes in shared memory.
     */
    DINLINE void init(const DataSpace<simDim>& blockCell, const int& linearThreadIdx, const DataSpace<simDim>& localCellOffset)
    {
        /* caching of E and B fields */
        cachedB = CachedBox::create < 0, ValueType_B > (BlockArea());
        cachedE = CachedBox::create < 1, ValueType_E > (BlockArea());

        /* instance of nvidia assignment operator */
        nvidia::functors::Assign assign;
        /* copy fields from global to shared */
        PMACC_AUTO(fieldBBlock, bBox.shift(blockCell));
        ThreadCollective<BlockArea> collective(linearThreadIdx);
        collective(
                  assign,
                  cachedB,
                  fieldBBlock
                  );
        /* copy fields from global to shared */
        PMACC_AUTO(fieldEBlock, eBox.shift(blockCell));
        collective(
                  assign,
                  cachedE,
                  fieldEBlock
                  );

        /* wait for shared memory to be initialized */
        __syncthreads();

        /* initialize random number generator with the local cell index in the simulation */
        this->randomGen.init(localCellOffset);
    }

    DINLINE float_X emission_prob(
        const float_X delta,
        const float_X chi,
        const float_X gamma,
        const float_X mass,
        const float_X charge) const
    {

        if(enableQEDTerm)
        {
            // quantum
            const float_X z = float_X(2.0/3.0) * delta / ((float_X(1.0) - delta) * chi);

            return DELTA_T * charge*charge * mass*SPEED_OF_LIGHT / (float_X(4.0)*PI*EPS0 * HBAR*HBAR) *
                float_X(1.732050807) / (float_X(2.0) * PI) * chi/gamma * (float_X(1.0) - delta) / delta *
                (this->curF_1[z] + float_X(1.5) * delta * chi * z * this->curF_2[z]);
        }
        else
        {
            // classical
            const float_X z = float_X(2.0/3.0) * delta / chi;

            return DELTA_T * charge*charge * mass*SPEED_OF_LIGHT / (float_X(4.0)*PI*EPS0 * HBAR*HBAR) *
                float_X(1.732050807) / (float_X(2.0) * PI) * chi/gamma / delta * this->curF_1[z];
        }
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

        /* particle position, used for field-to-particle interpolation */
        floatD_X pos = particle[position_];
        const int particleCellIdx = particle[localCellIdx_];
        /* multi-dim coordinate of the local cell inside the super cell */
        DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));
        /* interpolation of E- */
        const fieldSolver::numericalCellType::traits::FieldPosition<FieldE> fieldPosE;
        ValueType_E fieldE = Field2ParticleInterpolation()
            (cachedE.shift(localCell).toCursor(), pos, fieldPosE());
        /*                     and B-field on the particle position */
        const fieldSolver::numericalCellType::traits::FieldPosition<FieldB> fieldPosB;
        ValueType_B fieldB = Field2ParticleInterpolation()
            (cachedB.shift(localCell).toCursor(), pos, fieldPosB());

        const float3_X mom = particle[momentum_] / particle[weighting_];
        const float_X mom2 = math::dot(mom, mom);
        const float3_X mom_norm = mom * math::rsqrt(mom2);
        const float_X mass = frame::getMass<FrameType>();

        const float_X gamma = Gamma<>()(mom, mass);
        const float3_X vel = mom / (gamma * mass); // low accuracy?

        const float3_X lorentz = fieldE + math::cross(vel, fieldB);
        const float_X lorentz2 = math::dot(lorentz, lorentz);
        const float_X fieldE_long = math::dot(mom_norm, fieldE);

        const float_X H_eff = math::sqrt(lorentz2 - fieldE_long*fieldE_long);

        const float_X charge = math::abs(frame::getCharge<FrameType>());

        // Schwinger limit, unit: V/m
        const float_X c = SPEED_OF_LIGHT;
        const float_X E_S = mass*mass * c*c*c / (charge * HBAR);
        const float_X chi = gamma * H_eff / E_S;

        const float_X z = this->randomGen();
        if(z == float_X(0.0) || z == float_X(1.0))
            return 0;

        const float_X x = emission_prob_modified(z, chi, gamma, mass, charge);

        if(this->randomGen() < x)
        {
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

} // namespace synchrotronPhotons
} // namespace particles
} // namespace picongpu
