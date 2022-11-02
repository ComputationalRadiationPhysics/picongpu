/* Copyright 2016-2022 Heiko Burau
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

#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/fields/CellType.hpp"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/traits/frame/GetCharge.hpp"
#include "picongpu/traits/frame/GetMass.hpp"

#include <pmacc/algorithms/math/defines/cross.hpp>
#include <pmacc/algorithms/math/defines/dot.hpp>
#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/particles/operations/Assign.hpp>
#include <pmacc/particles/operations/Deselect.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace bremsstrahlung
        {
            template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
            Bremsstrahlung<T_IonSpecies, T_ElectronSpecies, T_PhotonSpecies>::Bremsstrahlung(
                const ScaledSpectrum::LookupTableFunctor& scaledSpectrumFunctor,
                const ScaledSpectrum::LookupTableFunctor& stoppingPowerFunctor,
                const GetPhotonAngle::GetPhotonAngleFunctor& getPhotonAngleFunctor,
                const uint32_t currentStep)
                : scaledSpectrumFunctor(scaledSpectrumFunctor)
                , stoppingPowerFunctor(stoppingPowerFunctor)
                , getPhotonAngleFunctor(getPhotonAngleFunctor)
                , photonMom(float3_X::create(0))
                , randomGen(RNGFactory::createRandom<Distribution>())
            {
                DataConnector& dc = Environment<>::get().DataConnector();

                /* initialize pointers on host-side tmp-field databoxes */
                auto fieldIonDensity = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                /* reset values to zero */
                fieldIonDensity->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));

                /* load species without copying the particle data to the host */
                auto ionSpecies = dc.get<T_IonSpecies>(T_IonSpecies::FrameType::getName(), true);

                /* compute ion density */
                using DensitySolver = typename particleToGrid::
                    CreateFieldTmpOperation<T_IonSpecies, particleToGrid::derivedAttributes::Density>::type::Solver;
                fieldIonDensity->template computeValue<CORE + BORDER, DensitySolver>(*ionSpecies, currentStep);

                /* initialize device-side tmp-field databoxes */
                this->ionDensityBox = fieldIonDensity->getDeviceDataBox();
            }

            template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
            template<typename T_Worker>
            DINLINE void Bremsstrahlung<T_IonSpecies, T_ElectronSpecies, T_PhotonSpecies>::collectiveInit(
                const T_Worker& worker,
                const DataSpace<simDim>& blockCell)
            {
                /* caching of ion density field */
                cachedIonDensity = CachedBox::create<0, ValueTypeIonDensity>(worker, BlockArea());

                /* instance of nvidia assignment operator */
                pmacc::math::operation::Assign assign;
                /* copy fields from global to shared */
                const auto fieldIonDensityBlock = ionDensityBox.shift(blockCell);

                auto collective = makeThreadCollective<BlockArea>();
                collective(worker, assign, cachedIonDensity, fieldIonDensityBlock);

                /* wait for shared memory to be initialized */
                worker.sync();
            }

            template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
            template<typename T_Worker>
            DINLINE void Bremsstrahlung<T_IonSpecies, T_ElectronSpecies, T_PhotonSpecies>::init(
                T_Worker const& worker,
                const DataSpace<simDim>& blockCell,
                const DataSpace<simDim>& localCellOffset)
            {
                /* initialize random number generator with the local cell index in the simulation */
                this->randomGen.init(localCellOffset);
            }


            template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
            template<typename T_Worker>
            DINLINE float3_X Bremsstrahlung<T_IonSpecies, T_ElectronSpecies, T_PhotonSpecies>::scatterByTheta(
                const T_Worker& worker,
                const float3_X vec,
                const float_X theta)
            {
                using namespace pmacc::algorithms;

                float_X sinTheta, cosTheta;
                pmacc::math::sincos(theta, sinTheta, cosTheta);

                const float_X phi = -pmacc::math::Pi<float_X>::value
                    + pmacc::math::Pi<float_X>::doubleValue * this->randomGen(worker);
                float_X sinPhi, cosPhi;
                pmacc::math::sincos(phi, sinPhi, cosPhi);

                const float3_X vecUp(0.0, 0.0, 1.0);
                float3_X vecOrtho1 = pmacc::math::cross(vecUp, vec);
                const float_X vecOrtho1Abs = math::abs(vecOrtho1);

                float3_X vecOrtho1_norm;
                if(vecOrtho1Abs == float_X(0.0))
                    vecOrtho1_norm = float3_X(1.0, 0.0, 0.0);
                else
                    vecOrtho1_norm = vecOrtho1 / vecOrtho1Abs;
                const float3_X vecOrtho2 = pmacc::math::cross(vecOrtho1_norm, vec);
                vecOrtho1 = vecOrtho1_norm * math::abs(vec);

                return vec * cosTheta + vecOrtho1 * (sinTheta * cosPhi) + vecOrtho2 * (sinTheta * sinPhi);
            }

            template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
            template<typename T_Worker>
            DINLINE unsigned int Bremsstrahlung<T_IonSpecies, T_ElectronSpecies, T_PhotonSpecies>::numNewParticles(
                const T_Worker& worker,
                FrameType& sourceFrame,
                int localIdx)
            {
                using namespace pmacc::algorithms;

                auto particle = sourceFrame[localIdx];

                /* particle position, used for field-to-particle interpolation */
                const floatD_X pos = particle[position_];
                const int particleCellIdx = particle[localCellIdx_];
                /* multi-dim coordinate of the local cell inside the super cell */
                const DataSpace<TVec::dim> localCell(
                    DataSpaceOperations<TVec::dim>::template map<TVec>(particleCellIdx));
                /* interpolation of fieldTmp */
                const picongpu::traits::FieldPosition<fields::CellType, FieldTmp, simDim> fieldTmpPos;
                const ValueTypeIonDensity ionDensity_norm
                    = Field2ParticleInterpolation()(cachedIonDensity.shift(localCell).toCursor(), pos, fieldTmpPos());

                /* TODO: obtain the ion density from the molare ion density in order to avoid the rescaling.
                 * So this should be: ionDensity = ionMolDensity / UNIT_AMOUNT_SUBSTANCE */
                const float_X ionDensity
                    = ionDensity_norm.x() * static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);

                const float_X weighting = particle[weighting_];

                const float_X c = SPEED_OF_LIGHT;
                float3_X mom = particle[momentum_] / weighting;
                const float_X momAbs = math::abs(mom);
                float3_X mom_norm = mom / momAbs;

                const float_X mass = frame::getMass<FrameType>();
                const float_X Ekin = (Gamma<>()(mom, mass) - float_X(1.0)) * mass * c * c;
                if(Ekin < electron::MIN_ENERGY)
                    return 0;

                /* electron deflection due to Rutherford scattering without modifying the electron
                   energy based on radiation emission */
                const float_X zMin
                    = float_X(1.0) / (pmacc::math::Pi<float_X>::value * pmacc::math::Pi<float_X>::value);
                const float_X zMax = float_X(1.0) / (electron::MIN_THETA * electron::MIN_THETA);
                const float_X z = zMin + this->randomGen(worker) * (zMax - zMin);
                const float_X theta = math::rsqrt(z);
                const float_X targetZ = GetAtomicNumbers<T_IonSpecies>::type::numberOfProtons;
                const float_X rutherfordCoeff = float_X(2.0) * ELECTRON_CHARGE * ELECTRON_CHARGE
                    / (float_X(4.0) * pmacc::math::Pi<float_X>::value * EPS0) * targetZ / Ekin;
                const float_X scaledDeflectionDCS
                    = pmacc::math::Pi<float_X>::value * (zMax - zMin) * rutherfordCoeff * rutherfordCoeff;
                const float_X deflectionProb = ionDensity * c * DELTA_T * scaledDeflectionDCS;

                if(this->randomGen(worker) < deflectionProb)
                {
                    mom = this->scatterByTheta(worker, mom, theta);
                    mom_norm = mom / momAbs;
                }

                /* non-radiative Bremsstrahlung */
                const float_X kappaCutoff = math::min(photon::SOFT_PHOTONS_CUTOFF / Ekin, float_X(1.0));
                const float_X stoppingPower = ionDensity * c * this->stoppingPowerFunctor(Ekin, kappaCutoff);
                const float_X newEkin = math::max(Ekin - stoppingPower * DELTA_T, float_X(0.0));
                const float_X newEkin_norm = newEkin / (mass * c * c);
                /* This is based on: (p / mc)^2 = (E_kin / mc^2)^2 + 2 * (E_kin / mc^2) */
                const float_X newMomAbs
                    = mass * c * math::sqrt(newEkin_norm * newEkin_norm + float_X(2.0) * newEkin_norm);
                const float_X deltaMom = newMomAbs - momAbs;
                particle[momentum_] = (mom + deltaMom * mom_norm) * weighting;

                /* photon emission */
                const float_X delta = this->randomGen(worker);
                const float_X kappa = math::pow(kappaCutoff, delta);
                const float_X scalingFactor = -math::log(kappaCutoff);
                const float_X emissionProb = photon::WEIGHTING_RATIO * scalingFactor * ionDensity * c * DELTA_T
                    * this->scaledSpectrumFunctor(Ekin, kappa);

                // raise a warning if the emission probability is too high.
                if(picLog::log_level & picLog::CRITICAL::lvl)
                {
                    if(emissionProb > float_X(photon::SINGLE_EMISSION_PROB_LIMIT))
                    {
                        const float_X Ekin_SI = Ekin * UNIT_ENERGY;
                        printf(
                            "[Bremsstrahlung] warning: emission probability is too high: \
                    p = %g, at Ekin = %g keV, kappa = %g, ion density = %g m^-3\n",
                            emissionProb,
                            Ekin_SI * UNITCONV_Joule_to_keV,
                            kappa,
                            ionDensity / (UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH));
                    }
                }

                if(this->randomGen(worker) < emissionProb)
                {
                    const float_X photonEnergy = kappa * Ekin;
                    this->photonMom = mom_norm * weighting / photon::WEIGHTING_RATIO * photonEnergy / c;
                    return 1;
                }

                return 0;
            }


            template<typename T_IonSpecies, typename T_ElectronSpecies, typename T_PhotonSpecies>
            template<typename Electron, typename Photon, typename T_Worker>
            DINLINE void Bremsstrahlung<T_IonSpecies, T_ElectronSpecies, T_PhotonSpecies>::operator()(
                const T_Worker& worker,
                Electron& electron,
                Photon& photon)
            {
                auto destPhoton
                    = pmacc::particles::operations::deselect<boost::mpl::vector<multiMask, momentum, weighting>>(
                        photon);

                namespace parOp = pmacc::particles::operations;
                parOp::assign(destPhoton, parOp::deselect<particleId>(electron));

                const float3_X elMom = electron[momentum_];
                const float_X weighting = electron[weighting_] / photon::WEIGHTING_RATIO;
                electron[momentum_] = elMom - this->photonMom; // ultra relativistic limit in terms of energy

                /* photon emission angle */
                const float_X mass = frame::getMass<FrameType>();
                const float_X gamma = Gamma<>()(elMom / weighting, mass);

                const float_X theta = this->getPhotonAngleFunctor(this->randomGen(worker), gamma);

                const float3_X scatteredPhotonMom = this->scatterByTheta(worker, this->photonMom, theta);

                photon[multiMask_] = 1;
                photon[momentum_] = scatteredPhotonMom;
                photon[weighting_] = weighting;
            }


        } // namespace bremsstrahlung
    } // namespace particles
} // namespace picongpu
