/** Copyright 2024 Filip Optolowicz
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

//! used in struct SynchrotronIdea
#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/param/synchrotron.param"

/** @file AlgorithmSynchrotron.hpp
 *
 * Synchrotron ALGORITHM for the Synchrotron model
 * Algorithm from the paper: "Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
 * developments" by A. Gonoskov et.Al.
 *
 * This file was created based on AlgorithmIonization.hpp
 *
 * This synchrotron extension consists of three new files:
 *      - synchrotron.param:            Contains the parameters for the synchrotron model and initialization
 *      - AlgorithmSynchrotron.hpp:     Defines the algorithm and data structures for the synchrotron model
 *      - SynchrotronRadiation.hpp:     Initializes the class with precomputed F1 and F2 functions

 * file locations:
 * picongpu/include/picongpu/param/synchrotron.param
 * picongpu/include/picongpu/particles/synchrotron/AlgorithmSynchrotron.hpp
 * picongpu/include/picongpu/simulation/stage/SynchrotronRadiation.hpp
 */
namespace picongpu
{
    namespace particles
    {
        namespace synchrotron
        {
            /** SynchrotronIdea
             * Main algorithm of the synchrotron radiation model.
             * From paper: "Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
             * developments" by A. Gonoskov et.Al.
             */
            struct SynchrotronIdea
            {
                //! logInterpolation is a helper function for the logarithmic interpolation
                HDINLINE void logInterpolation(
                    int16_t const index,
                    float_X& F,
                    float_X& Ftemp,
                    float_X const f,
                    int const i,
                    GridBuffer<float_X, 2>::DataBoxType F1F2DeviceBuff) const
                {
                    F = F1F2DeviceBuff(DataSpace<2>{index, i});
                    Ftemp = F1F2DeviceBuff(DataSpace<2>{index + 1, i});
                    //! F1 = F1**((1-f)) * Ftemp**f
                    F = math::pow(F, 1._X - f) * math::pow(Ftemp, f);
                }

                /** Functor implementation
                 * @tparam EType type of electric field
                 * @tparam BType type of magnetic field
                 * @tparam ParticleType type of electron particle
                 *
                 * @param bField magnetic field value at t=0
                 * @param eField electric field value at t=0
                 * @param parentElectron instance of electron with position at t=0 and momentum at t=-1/2
                 * @param randNr random number, equally distributed in range [0.:1.0]
                 *
                 * @return photon energy with respect to electron energy if photon is created 0 otherwise
                 */

                template<typename T_Worker, typename EType, typename BType, typename ParticleType>
                HDINLINE float_X operator()(
                    const T_Worker& worker,
                    const BType bField,
                    const EType eField,
                    ParticleType& parentElectron,
                    float_64 randNr1,
                    float_64 randNr2,
                    GridBuffer<float_X, 2>::DataBoxType F1F2DeviceBuff,
                    GridBuffer<int32_t, 1>::DataBoxType failedRequirementQBuff) const
                {
                    constexpr float_64 minZqExponent = params::InterpolationParams::minZqExponent;
                    constexpr float_64 maxZqExponent = params::InterpolationParams::maxZqExponent;
                    constexpr float_X interpolationPoints
                        = static_cast<float_X>(params::InterpolationParams::numberTableEntries - 1);
                    constexpr float_X stepWidthLogatihmicScale
                        = (static_cast<float_X>(maxZqExponent - minZqExponent) / interpolationPoints);

                    /** zq = 2/3 * chi^(-1) * delta / (1 - delta) ;
                     *     chi - ratio of the typical photon energy to the electron kinetic energy:
                     *          if larger than 1 the radiation is in the quantum regime
                     *     delta - the ratio of the photon energy to the electron energy
                     * see Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
                     * developments, A. Gonoskov et al. */

                    //! delta = randNr1^3 but we use randNr1^2 later so we calculate it here
                    float_X const r1r1 = randNr1 * randNr1;
                    float_X const delta = r1r1 * randNr1;

                    //! calculate Heff
                    float_X const mass = attribute::getMass(parentElectron[weighting_], parentElectron);
                    float3_X const vel = Velocity()(parentElectron[momentum_], mass);

                    float_X const Vmag = pmacc::math::l2norm(vel);
                    float3_X const crossVB = pmacc::math::cross(vel, bField);
                    //! Velocity normalized
                    float3_X const Vnorm = vel / Vmag;
                    float_X const dotVnormE = pmacc::math::dot(Vnorm, eField);
                    float3_X const eFieldPlusCrossVB = eField + crossVB;

                    float_X HeffValue = pmacc::math::dot(eFieldPlusCrossVB, eFieldPlusCrossVB) - dotVnormE * dotVnormE;
                    /* this check is important to avoid numerical rounding errors that result in NaN
                     *  when taking a sqrt
                     */
                    if(HeffValue <= 0)
                    {
                        return 0;
                    }
                    HeffValue = math::sqrt(HeffValue);


                    //! Calculate chi
                    float_X const gamma = Gamma()(parentElectron[momentum_], mass);
                    float_X const inverse_E_Schwinger = (-sim.pic.getElectronCharge() * HBAR)
                        / (ELECTRON_MASS * ELECTRON_MASS * sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight()
                           * sim.pic.getSpeedOfLight());
                    float_X const chi = HeffValue * gamma * inverse_E_Schwinger;

                    //! zq
                    const float_64 oneMinusDeltaOverDelta = (1. - delta) / delta;
                    const float_X zq = 2._X / (3._X * chi) / oneMinusDeltaOverDelta;

                    //! zq convert to index and F1 and F2
                    const float_X zqExponent = math::log2(zq);
                    const int16_t index
                        = static_cast<int16_t>((zqExponent - minZqExponent) / stepWidthLogatihmicScale);

                    float_X F1 = 0._X;
                    float_X F2 = 0._X;
                    float_X Ftemp = 0._X;

                    if(index >= 0 && index < interpolationPoints - 2)
                    {
                        float_X zq1 = math::pow(2, minZqExponent + stepWidthLogatihmicScale * (index));
                        float_X zq2 = math::pow(2, minZqExponent + stepWidthLogatihmicScale * (index + 1));
                        float_X f = (zq - zq1) / (zq2 - zq1);

                        /** @todo Test Logarithmic interpolation:
                         *     1) how slow is it?
                         *     2) how much more accurate it is? (propably not much)
                         *     3) if not worth to do it replace with:
                         *         F1 = F1F2DeviceBuff(DataSpace<2>{index, 0});
                         *         F2 = F1F2DeviceBuff(DataSpace<2>{index, 1});
                         */

                        //! logInterpolate F1 and F2 -> passed by reference
                        logInterpolation(index, F1, Ftemp, f, 0, F1F2DeviceBuff);
                        logInterpolation(index, F2, Ftemp, f, 1, F1F2DeviceBuff);
                    }

                    /** Calculating the numeric factor:
                     *      dt * (e**2 * m_e * c /( hbar**2 * eps0 * 4 * np.pi))
                     *   == dt * (e/hbar * e/hbar * m_e * c /( eps0 * 4 * np.pi)) and avoid floating point precision
                     * overflows
                     */
                    float_X numericFactor = sim.pic.getDt() * sim.pic.getElectronCharge() / HBAR
                        * sim.pic.getElectronCharge() / HBAR
                        * ELECTRON_MASS * sim.pic.getSpeedOfLight() / (EPS0 * 4._X * PI);

                    if constexpr(params::supressRequirementWarning == false)
                    {
                        float_X requirement1 = numericFactor * 1.5_X * math::pow(chi, 2._X / 3._X) / gamma;
                        float_X requirement2 = numericFactor * 0.5_X * chi / gamma;

                        /** this warning means that the propability of generating a photon is high for given dt (higher
                         * than 10%) this means that we possibly generate photons every timestep and
                         * underestimate the radiation. The solution would be to implement substepping.
                         * If requirement is not met send warning at every iteration
                         */
                        if(requirement1 > 0.1_X || requirement2 > 0.1_X)
                            alpaka::atomicExch(worker.getAcc(), &(failedRequirementQBuff(DataSpace<1>{0})), 1);
                    }

                    numericFactor *= math::sqrt(3._X) / (pmacc::math::Pi<float_X>::doubleValue);

                    //! Calculate propability:
                    float_X const numerator
                        = oneMinusDeltaOverDelta * chi * (F1 + 3._X * delta * zq * chi / 2._X * F2);
                    //! the delta is in oneMinusDeltaOverDelta in numerator1
                    float_X const denominator = gamma;

                    float_X propability = numericFactor * numerator / denominator * 3._X * r1r1;

                    return (propability > randNr2) * delta;
                }
            };

            /** \struct AlgorithmSynchrotron
             * This struct is passed to creation::createParticlesFromSpecies in the file:
             * include/picongpu/particles/ParticlesFunctors.hpp
             *
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the photon species to be created
             * @tparam T_SrcSpecies  type or name as PMACC_CSTRING of the particle species that radiates so electrons
             * only
             *
             * Takes care of:
             *  - random number generation
             *  - E and B field interpolation
             *  - maby something else as well
             */
            template<typename T_SrcSpecies, typename T_DestSpecies>
            struct AlgorithmSynchrotron
            {
                using DestSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_DestSpecies>;
                using SrcSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SrcSpecies>;

                using FrameType = typename SrcSpecies::FrameType;

                /** specify field to particle interpolation scheme */
                using Field2ParticleInterpolation = typename pmacc::traits::Resolve<
                    typename pmacc::traits::GetFlagType<FrameType, interpolation<>>::type>::type;

                /** margins around the supercell for the interpolation of the field on the cells */
                using LowerMargin = typename GetMargin<Field2ParticleInterpolation>::LowerMargin;
                using UpperMargin = typename GetMargin<Field2ParticleInterpolation>::UpperMargin;

                /** relevant area of a block */
                using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

                BlockArea BlockDescription;

            private:
                /** random number generator */
                using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                using Distribution = pmacc::random::distributions::Uniform<float_64>;
                using RandomGen = typename RNGFactory::GetRandomType<Distribution>::type;
                RandomGen randomGen;

                using TVec = MappingDesc::SuperCellSize;

                using ValueType_E = FieldE::ValueType;
                using ValueType_B = FieldB::ValueType;
                /** global memory EM-field and current density device databoxes */
                PMACC_ALIGN(eBox, FieldE::DataBoxType);
                PMACC_ALIGN(bBox, FieldB::DataBoxType);
                /** shared memory EM-field device databoxes */
                PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize, 1>>);
                PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize, 0>>);

                //! F1F2DeviceBuff_ is a pointer to a databox containing F1 and F2 values. m stands for member
                GridBuffer<float_X, 2>::DataBoxType m_F1F2DeviceBuff;
                GridBuffer<int32_t, 1>::DataBoxType m_failedRequirementQBuff;

            public:
                /** host constructor initializing member : random number generator */
                AlgorithmSynchrotron(
                    const uint32_t currentStep,
                    GridBuffer<float_X, 2>::DataBoxType F1F2DeviceBuff,
                    GridBuffer<int32_t, 1>::DataBoxType failedRequirementQBuff)
                    : randomGen(RNGFactory::createRandom<Distribution>())
                    , m_F1F2DeviceBuff{F1F2DeviceBuff}
                    , m_failedRequirementQBuff{failedRequirementQBuff}
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    /** initialize pointers on host-side E-(B-)field and current density databoxes */
                    auto fieldE = dc.get<FieldE>(FieldE::getName());
                    auto fieldB = dc.get<FieldB>(FieldB::getName());
                    /** initialize device-side E-(B-)field and current density databoxes */
                    eBox = fieldE->getDeviceDataBox();
                    bBox = fieldB->getDeviceDataBox();
                }

                /** cache fields used by this functor
                 *
                 * @warning this is a collective method and calls synchronize
                 *
                 * @tparam T_Worker lockstep::Worker, lockstep worker
                 *
                 * @param worker lockstep worker
                 * @param blockCell relative offset (in cells) to the local domain plus the guarding cells
                 * @param workerCfg configuration of the worker
                 */
                template<typename T_Worker>
                DINLINE void collectiveInit(const T_Worker& worker, const DataSpace<simDim>& blockCell)
                {
                    /** caching of E and B fields */
                    cachedB = CachedBox::create<0, ValueType_B>(worker, BlockArea());
                    cachedE = CachedBox::create<1, ValueType_E>(worker, BlockArea());

                    /** instance of alpaka assignment operator */
                    pmacc::math::operation::Assign assign;
                    /** copy fields from global to shared */
                    auto fieldBBlock = bBox.shift(blockCell);
                    auto collective = makeThreadCollective<BlockArea>();
                    collective(worker, assign, cachedB, fieldBBlock);
                    /** copy fields from global to shared */
                    auto fieldEBlock = eBox.shift(blockCell);
                    collective(worker, assign, cachedE, fieldEBlock);

                    /** wait for shared memory to be initialized */
                    worker.sync();
                }

                /** Initialization function on device
                 *
                 * \brief Cache EM-fields on device
                 *         and initialize possible prerequisites for radiation, like e.g. random number generator.
                 *
                 * This function will be called inline on the device which must happen BEFORE threads diverge
                 * during loop execution. The reason for this is the `cupla::__syncthreads( acc )` call which is
                 * necessary after initializing the E-/B-field shared boxes in shared memory.
                 */
                template<typename T_Worker>
                DINLINE void init(
                    T_Worker const&,
                    const DataSpace<simDim>& localSuperCellOffset,
                    const uint32_t rngIdx)

                {
                    auto rngOffset = DataSpace<simDim>::create(0);
                    rngOffset.x() = rngIdx;
                    auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
                    numRNGsPerSuperCell.x() = FrameType::frameSize;
                    this->randomGen.init(localSuperCellOffset * numRNGsPerSuperCell + rngOffset);
                }

                /** Determine number of new macro photons due to radiation -> called by CreationKernel
                 *
                 * @param electronFrame reference to frame of the electron
                 * @param localIdx local (linear) index in super cell / frame
                 */
                template<typename T_Worker>
                DINLINE uint32_t numNewParticles(const T_Worker& worker, FrameType& electronFrame, int localIdx)
                {
                    /** alias for the single macro-particle - electron */
                    auto particle = electronFrame[localIdx];
                    /** particle position, used for field-to-particle interpolation */
                    floatD_X pos = particle[position_];
                    const int particleCellIdx = particle[localCellIdx_];
                    /** multi-dim coordinate of the local cell inside the super cell */
                    DataSpace<TVec::dim> localCell = pmacc::math::mapToND(TVec::toRT(), particleCellIdx);
                    /** interpolation of E-... */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldE> fieldPosE;
                    ValueType_E eField = Field2ParticleInterpolation()(cachedE.shift(localCell), pos, fieldPosE());
                    /**                     ...and B-field on the particle position */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldB> fieldPosB;
                    ValueType_B bField = Field2ParticleInterpolation()(cachedB.shift(localCell), pos, fieldPosB());

                    //! use the algorithm from the SynchrotronIdea struct
                    SynchrotronIdea synchrotronAlgo;
                    float_X photonEnergy = synchrotronAlgo(
                        worker,
                        bField,
                        eField,
                        particle,
                        this->randomGen(worker),
                        this->randomGen(worker),
                        m_F1F2DeviceBuff,
                        m_failedRequirementQBuff);


                    //! conversion factor from photon energy to momentum
                    constexpr float_X convFactor = 1.0_X / sim.pic.getSpeedOfLight();
                    //! if this is wrong uncomment the lines below and comment this line
                    float3_X const PhotonMomentum = particle[momentum_] * photonEnergy * convFactor;

                    //! save to member variable to use in creation of new photon
                    m_PhotonMomentum = PhotonMomentum;

                    //! generate one photon if energy is > minEnergy
                    return (pmacc::math::l2norm(PhotonMomentum) > params::minEnergy * convFactor);
                }

                /** Functor implementation
                 *
                 * Ionization model specific particle creation
                 *
                 * @tparam T_parentElectron type of ion species that is radiating
                 * @tparam T_childPhoton type of Photon species that is created
                 * @param parentElectron electron instance that radiates
                 * @param childPhoton Photon instance that is created
                 */
                template<typename T_parentElectron, typename T_childPhoton, typename T_Worker>
                DINLINE void operator()(
                    const T_Worker& worker,
                    IdGenerator& idGen,
                    T_parentElectron& parentElectron,
                    T_childPhoton& childPhoton)
                {
                    /** for not mixing operations::assign up with the nvidia functor assign */
                    namespace partOp = pmacc::particles::operations;
                    /** each thread sets the multiMask hard on "particle" (=1) */
                    childPhoton[multiMask_] = 1u;

                    /** each thread initializes a clone of the parent Electron but leaving out
                     * some attributes:
                     * - multiMask: reading from global memory takes longer than just setting it again explicitly
                     * - momentum: because the Photon would get a higher energy because of the Electron mass
                     */
                    auto targetPhotonClone = partOp::deselect<pmacc::mp_list<multiMask, momentum>>(childPhoton);

                    targetPhotonClone.copyAndInit(worker, idGen, partOp::deselect<particleId>(parentElectron));

                    childPhoton[momentum_] = m_PhotonMomentum;

                    /** conservatElectron of momentum */
                    if constexpr(params::ElectronRecoil)
                        parentElectron[momentum_] -= m_PhotonMomentum;
                }

            private:
                //! energy of emitted photon with respect to electron energy
                float3_X m_PhotonMomentum;

            }; // end of struct AlgorithmSynchrotron
        } // namespace synchrotron
    } // namespace particles
} // namespace picongpu
