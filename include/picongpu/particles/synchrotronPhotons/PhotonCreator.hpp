/* Copyright 2015-2021 Heiko Burau
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

#include "SynchrotronFunctions.hpp"
#include "picongpu/algorithms/Gamma.hpp"
#include <pmacc/algorithms/math/defines/dot.hpp>
#include <pmacc/algorithms/math/defines/cross.hpp>
#include "picongpu/traits/frame/GetMass.hpp"
#include "picongpu/traits/frame/GetCharge.hpp"
#include <pmacc/particles/operations/Assign.hpp>
#include <pmacc/particles/operations/Deselect.hpp>
#include <pmacc/particles/traits/ResolveAliasFromSpecies.hpp>
#include "picongpu/fields/CellType.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/traits/FieldPosition.hpp"

#include <pmacc/random/methods/methods.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/RNGProvider.hpp>

#include <pmacc/traits/Resolve.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <pmacc/meta/conversion/TypeToPointerPair.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace synchrotronPhotons
        {
            /** Functor creating photons from electrons according to synchrotron radiation.
             *
             * The numerical model is taken from:
             *
             * Gonoskov, A., et al. "Extended particle-in-cell schemes for physics
             * in ultrastrong laser fields: Review and developments."
             * Physical Review E 92.2 (2015): 023305.
             *
             * This functor is called by the general particle creation module.
             *
             * \tparam T_ElectronSpecies
             * \tparam T_PhotonSpecies
             */
            template<typename T_ElectronSpecies, typename T_PhotonSpecies>
            struct PhotonCreator
            {
                using ElectronSpecies = T_ElectronSpecies;
                using PhotonSpecies = T_PhotonSpecies;

                using FrameType = typename ElectronSpecies::FrameType;

                /* specify field to particle interpolation scheme */
                using Field2ParticleInterpolation =
                    typename pmacc::particles::traits::ResolveAliasFromSpecies<ElectronSpecies, interpolation<>>::type;

                /* margins around the supercell for the interpolation of the field on the cells */
                using LowerMargin = typename GetMargin<Field2ParticleInterpolation>::LowerMargin;
                using UpperMargin = typename GetMargin<Field2ParticleInterpolation>::UpperMargin;

                /* relevant area of a block */
                using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

                BlockArea BlockDescription;

                using TVec = MappingDesc::SuperCellSize;

                using ValueType_E = FieldE::ValueType;
                using ValueType_B = FieldB::ValueType;

            private:
                /* global memory EM-field device databoxes */
                PMACC_ALIGN(eBox, FieldE::DataBoxType);
                PMACC_ALIGN(bBox, FieldB::DataBoxType);
                /* shared memory EM-field device databoxes */
                PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize, 1>>);
                PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize, 0>>);

                PMACC_ALIGN(curF_1, SynchrotronFunctions::SyncFuncCursor);
                PMACC_ALIGN(curF_2, SynchrotronFunctions::SyncFuncCursor);

                PMACC_ALIGN(photon_mom, float3_X);

                /* random number generator */
                using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                using Distribution = pmacc::random::distributions::Uniform<float_X>;
                using RandomGen = typename RNGFactory::GetRandomType<Distribution>::type;
                RandomGen randomGen;

            public:
                /* host constructor initializing member : random number generator */
                PhotonCreator(
                    const SynchrotronFunctions::SyncFuncCursor& curF_1,
                    const SynchrotronFunctions::SyncFuncCursor& curF_2)
                    : curF_1(curF_1)
                    , curF_2(curF_2)
                    , photon_mom(float3_X::create(0))
                    , randomGen(RNGFactory::createRandom<Distribution>())
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    /* initialize pointers on host-side E-(B-)field databoxes */
                    auto fieldE = dc.get<FieldE>(FieldE::getName(), true);
                    auto fieldB = dc.get<FieldB>(FieldB::getName(), true);
                    /* initialize device-side E-(B-)field databoxes */
                    eBox = fieldE->getDeviceDataBox();
                    bBox = fieldB->getDeviceDataBox();
                }

                /** cache fields used by this functor
                 *
                 * @warning this is a collective method and calls synchronize
                 *
                 * @tparam T_Acc alpaka accelerator type
                 * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                 *
                 * @param acc alpaka accelerator
                 * @param blockCell relative offset (in cells) to the local domain plus the guarding cells
                 * @param workerCfg configuration of the worker
                 */
                template<typename T_Acc, typename T_WorkerCfg>
                DINLINE void collectiveInit(
                    const T_Acc& acc,
                    const DataSpace<simDim>& blockCell,
                    const T_WorkerCfg& workerCfg)
                {
                    /* caching of E and B fields */
                    cachedB = CachedBox::create<0, ValueType_B>(acc, BlockArea());
                    cachedE = CachedBox::create<1, ValueType_E>(acc, BlockArea());

                    /* instance of nvidia assignment operator */
                    nvidia::functors::Assign assign;
                    /* copy fields from global to shared */
                    auto fieldBBlock = bBox.shift(blockCell);
                    ThreadCollective<BlockArea, T_WorkerCfg::numWorkers> collective(workerCfg.getWorkerIdx());
                    collective(acc, assign, cachedB, fieldBBlock);
                    /* copy fields from global to shared */
                    auto fieldEBlock = eBox.shift(blockCell);
                    collective(acc, assign, cachedE, fieldEBlock);

                    /* wait for shared memory to be initialized */
                    cupla::__syncthreads(acc);
                }

                /** Initialization function on device
                 *
                 * \brief Cache EM-fields on device
                 *         and initialize possible prerequisites for ionization, like e.g. random number generator.
                 *
                 * This function will be called inline on the device which must happen BEFORE threads diverge
                 * during loop execution. The reason for this is the `cupla::__syncthreads( acc )` call which is
                 * necessary after initializing the E-/B-field shared boxes in shared memory.
                 */
                template<typename T_Acc>
                DINLINE void init(
                    T_Acc const& acc,
                    const DataSpace<simDim>& blockCell,
                    const int& linearThreadIdx,
                    const DataSpace<simDim>& localCellOffset)
                {
                    /* initialize random number generator with the local cell index in the simulation */
                    this->randomGen.init(localCellOffset);
                }

                /** Get the photon emission probability
                 *
                 * @param delta normalized (to the electron energy) photon energy
                 * @param chi quantum-nonlinearity parameter
                 * @param gamma electron gamma
                 */
                DINLINE float_X emission_prob(const float_X delta, const float_X chi, const float_X gamma) const
                {
                    // catch these special values because otherwise a NaN is returned whereas it should be a zero.
                    if(chi == float_X(0.0) || delta == float_X(0.0) || (float_X(1.0) - delta) == float_X(0.0))
                        return float_X(0.0);

                    const float_X mass = frame::getMass<FrameType>();
                    const float_X charge = frame::getCharge<FrameType>();

                    const float_X sqrtOf3 = 1.7320508075688772;
                    const float_X factor = DELTA_T * charge * charge * mass * SPEED_OF_LIGHT
                        / (float_X(4.0) * PI * EPS0 * HBAR * HBAR) * sqrtOf3 / (float_X(2.0) * PI) * chi / gamma;

                    if(enableQEDTerm)
                    {
                        // quantum
                        const float_X z = float_X(2.0 / 3.0) * delta / ((float_X(1.0) - delta) * chi);

                        return factor * (float_X(1.0) - delta) / delta
                            * (this->curF_1[z] + float_X(1.5) * delta * chi * z * this->curF_2[z]);
                    }
                    else
                    {
                        // classical
                        const float_X z = float_X(2.0 / 3.0) * delta / chi;

                        return factor / delta * this->curF_1[z];
                    }
                }

                /** Get the *scaled* photon emission probability
                 *
                 * The scaling avoids an infrared divergence.
                 *
                 * @param deltaScaled scaled and normalized (to the electron energy) photon energy
                 * @param chi quantum-nonlinearity parameter
                 * @param gamma electron gamma
                 */
                DINLINE float_X
                emission_prob_scaled(const float_X deltaScaled, const float_X chi, const float_X gamma) const
                {
                    const float_X delta = deltaScaled * deltaScaled * deltaScaled;
                    return float_X(3.0) * deltaScaled * deltaScaled * emission_prob(delta, chi, gamma);
                }

                /** Return the number of target particles to create from each source particle.
                 *
                 * Called for each frame of the source species.
                 *
                 * @param sourceFrame Frame of the source species
                 * @param localIdx Index of the source particle within frame
                 * @return number of particle to be created from each source particle
                 */
                template<typename T_Acc>
                DINLINE unsigned int numNewParticles(const T_Acc& acc, FrameType& sourceFrame, int localIdx)
                {
                    using namespace pmacc::algorithms;

                    auto particle = sourceFrame[localIdx];

                    /* particle position, used for field-to-particle interpolation */
                    const floatD_X pos = particle[position_];
                    const int particleCellIdx = particle[localCellIdx_];
                    /* multi-dim coordinate of the local cell inside the super cell */
                    DataSpace<TVec::dim> localCell(
                        DataSpaceOperations<TVec::dim>::template map<TVec>(particleCellIdx));
                    /* interpolation of E-field on the particle position */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldE> fieldPosE;
                    ValueType_E fieldE
                        = Field2ParticleInterpolation()(cachedE.shift(localCell).toCursor(), pos, fieldPosE());
                    /* interpolation of B-field on the particle position */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldB> fieldPosB;
                    ValueType_B fieldB
                        = Field2ParticleInterpolation()(cachedB.shift(localCell).toCursor(), pos, fieldPosB());

                    /* All computation below is in the single "real" particle picture.
                     * The macroparticle weighting factor is reintroduced at the end of this code block. */
                    const float3_X mom = particle[momentum_] / particle[weighting_];
                    const float_X mom2 = pmacc::math::dot(mom, mom);
                    const float3_X mom_norm = mom * math::rsqrt(mom2);
                    const float_X mass = frame::getMass<FrameType>();

                    const float_X gamma = Gamma<>()(mom, mass);
                    const float3_X vel = mom / (gamma * mass); // low accuracy?

                    const float3_X lorentzForceOverCharge = fieldE + pmacc::math::cross(vel, fieldB);
                    const float_X lorentzForceOverCharge2
                        = pmacc::math::dot(lorentzForceOverCharge, lorentzForceOverCharge);
                    const float_X fieldE_long = pmacc::math::dot(mom_norm, fieldE);

                    // effective magnetic strength (in cgs)
                    const float_X H_eff = math::sqrt(lorentzForceOverCharge2 - fieldE_long * fieldE_long);

                    const float_X charge = math::abs(frame::getCharge<FrameType>());

                    const float_X c = SPEED_OF_LIGHT;
                    // Schwinger limit, unit: V/m (in cgs)
                    const float_X E_S = mass * mass * c * c * c / (charge * HBAR);
                    // quantum-nonlinearity parameter
                    const float_X chi = gamma * H_eff / E_S;

                    const float_X deltaScaled = this->randomGen(acc);

                    const float_X x = emission_prob_scaled(deltaScaled, chi, gamma);

                    // raise a warning if the emission probability is too high.
                    if(picLog::log_level & picLog::CRITICAL::lvl)
                    {
                        if(x > float_X(SINGLE_EMISSION_PROB_LIMIT))
                        {
                            const float_X delta = deltaScaled * deltaScaled * deltaScaled;
                            printf(
                                "[SynchrotronPhotons] warning: emission probability is too high: p = %g, at delta = "
                                "%g, chi = %g, gamma = %g\n",
                                x,
                                delta,
                                chi,
                                gamma);
                        }
                    }

                    if(this->randomGen(acc) < x)
                    {
                        const float_X delta = deltaScaled * deltaScaled * deltaScaled;
                        const float_X photonMom_abs = delta * mass * c * gamma;
                        if(photonMom_abs > SOFT_PHOTONS_CUTOFF_MOM)
                        {
                            this->photon_mom = mom_norm * photonMom_abs * particle[weighting_];
                            return 1;
                        }
                    }

                    return 0;
                }

                /** Functor implementation: setting photon and electron properties
                 *
                 * Called once for each single particle creation.
                 *
                 * \tparam Electron type of electron which creates the photon
                 * \tparam Photon type of photon that is created
                 */
                template<typename Electron, typename Photon, typename T_Acc>
                DINLINE void operator()(const T_Acc& acc, Electron& electron, Photon& photon) const
                {
                    namespace parOp = pmacc::particles::operations;
                    auto destPhoton = parOp::deselect<boost::mpl::vector<multiMask, momentum>>(photon);
                    parOp::assign(destPhoton, parOp::deselect<particleId>(electron));

                    photon[multiMask_] = 1;
                    photon[momentum_] = this->photon_mom;
                    electron[momentum_] -= this->photon_mom;
                }
            };

        } // namespace synchrotronPhotons
    } // namespace particles
} // namespace picongpu
