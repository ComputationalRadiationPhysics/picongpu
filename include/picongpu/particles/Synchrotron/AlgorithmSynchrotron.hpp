/* Copyright 2014-2024 Marco Garten, Jakob Trojok, Filip Optolowicz
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

#include "picongpu/algorithms/KinEnergy.hpp" // used in struct SynchrotronIdea

/** @file AlgorithmSynchrotron.hpp
 *
 * Synchrotron ALGORITHM for the Synchrotron model
 * Algorithm from the paper: "Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
 * developments" by A. Gonoskov et.Al.
 *
 * This file was created based on AlgorithmIonization.hpp
 *
 * @todo make better desciptions of SynchrotronRadiation.hpp and ParticleFunctors.hpp
 * @todo remove copy of this from SynchrotronRadiation.hpp
 *
 * This synchrotron extension consists of two new files:
 *      - AlgorithmSynchrotron.hpp:     Defines the algorithm and data structures for the synchrotron model
 *      - SynchrotronRadiation.hpp:     Initializes the class with precomputed F1 and F2 functions
 * and modifies two existing files:
 *      - Simulation.hpp:               Registers the SynchrotronRadiation stage
 *      - ParticleFunctors.hpp:         Chooses which particles are affected by SynchrotronRadiation
 *
 * The call structure goes like this:
 *     Simulation.hpp -> SynchrotronRadiation.hpp -> ParticleFunctors.hpp -> AlgorithmSynchrotron.hpp
 *
 * @warning tested only on cpu -> check "m_normalizedPhotonEnergy" variable on GPU
 *
 */
namespace picongpu
{
    namespace particles
    {
        namespace synchrotron
        {
            ///@todo remove
            constexpr bool T_Debug = false;
            // Carefull -> prints out a lot of stuff. Bigger simulations will produce massive outputs

            /// to access params:
            /// picongpu::particles::synchrotron::params
            namespace params
            {
                // Turn off or turn on the electron recoil from electrons generated.
                constexpr bool ElectronRecoil = false;

                struct FirstSynchrotronFunctionParams // Parameters how to compute first synhrotron function
                {
                    static constexpr float_64 logEnd = 6.6; //! log2(100.0), arbitrary cutoff, for 2nd kind cyclic
                                                            //! bessel function -> function close enough to zero
                    static constexpr uint32_t numberSamplePoints = 1024u; // number of sample points to use in
                                                                          // integration in firstSynchrotronFunction
                };

                struct InterpolationParams // parameters of precomputation of interpolation table -> the table
                                           // "tableValuesF1F2" is in simulation/stage/SynchrotronRadiation.hpp
                {
                    static constexpr uint64_t numberTableEntries = 1024u; // number of synchrotron function values
                                                                          // to precompute and store in table
                    static constexpr float_64 minZqExponent = -32; // cutoff energy -> @todo: make this a parameter
                    static constexpr float_64 maxZqExponent = 3.3; // don't change. or change. but don't change.
                };

                enum struct Accessor : uint32_t // used for table access -> the table "tableValuesF1F2" is in
                                                // simulation/stage/SynchrotronRadiation.hpp
                {
                    f1 = 0u,
                    f2 = 1u
                };

                static constexpr uint32_t u32(Accessor const t)
                {
                    return static_cast<uint32_t>(t);
                }
            } // namespace params

            /** \struct SynchrotronIdea
             * Main algorithm of the synchrotron radiation model.
             * From paper: "Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
             * developments" by A. Gonoskov et.Al.
             */
            struct SynchrotronIdea
            {
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
                 * @return photon energy if photon is created 0 otherwise
                 */

// just a shortcut to access the tableValuesF1F2 (see: InterpolationParams)
#define F1F2(zq, i) F1F2DeviceBuff(DataSpace<2>{zq, i})

                template<typename EType, typename BType, typename ParticleType>
                HDINLINE float_X // return type -> energy of a photon relative to the electron energy
                operator()(
                    const BType bField,
                    const EType eField,
                    ParticleType& parentElectron,
                    float_X randNr1,
                    float_X randNr2,
                    GridBuffer<float_64, 2>::DataBoxType F1F2DeviceBuff) const
                {
                    constexpr int16_t minZqExponent = params::InterpolationParams::minZqExponent;
                    constexpr int16_t maxZqExponent = params::InterpolationParams::maxZqExponent;
                    constexpr float_64 interpolationPoints
                        = static_cast<float_64>(params::InterpolationParams::numberTableEntries);
                    constexpr float_64 stepWidthLogatihmicScale
                        = (static_cast<float_64>(maxZqExponent - minZqExponent) / interpolationPoints);

                    /// zq = 2/3 * chi^(-1) * delta / (1 - delta) ;
                    ///     chi - ratio of the typical photon energy to the electron kinetic energy
                    ///     delta - the ratio of the photon energy to the electron energy
                    /// see Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
                    /// developments, A. Gonoskov et.Al.

                    // delta
                    float_X const r1r1 = randNr1 * randNr1;
                    float_X const delta = r1r1 * randNr1;

                    // calculate Heff
                    float_X const mass = attribute::getMass(parentElectron[weighting_], parentElectron);
                    float3_X const vel = Velocity()(parentElectron[momentum_], mass);

                    float_X const Vmag = pmacc::math::l2norm(vel);
                    float3_X const crossVB = pmacc::math::cross(vel, bField);
                    float3_X const Vnorm = vel / Vmag; // Velocity normalized -> not L2Norm
                    float_X const dotVnormE = pmacc::math::dot(Vnorm, eField);
                    float3_X const eFieldPlusCrossVB = eField + crossVB;

                    float_X HeffValue = pmacc::math::dot(eFieldPlusCrossVB, eFieldPlusCrossVB) - dotVnormE * dotVnormE;
                    if(HeffValue <= 0)
                    {
                        return 0;
                    }
                    HeffValue = math::sqrt(HeffValue);


                    // Calculate chi
                    float_X const gamma = Gamma()(parentElectron[momentum_], mass);
                    float_X const chi = -ELECTRON_CHARGE * HBAR * HeffValue * gamma
                        / (ELECTRON_MASS * ELECTRON_MASS * SPEED_OF_LIGHT * SPEED_OF_LIGHT * SPEED_OF_LIGHT);

                    // zq
                    float_64 zq = 2 * delta / (3 * chi * (1 - delta));

                    // zq convert to index and F1 and F2
                    const float_64 zqExponent = math::log2(zq);
                    const int16_t index
                        = static_cast<int16_t>((zqExponent - minZqExponent) / stepWidthLogatihmicScale);

                    float_64 F1 = 0;
                    float_64 F2 = 0;
                    float_64 Ftemp = 0;
                    if(index >= 0 && index < interpolationPoints - 1)
                    {
                        float_64 zq1 = math::pow(2, minZqExponent + stepWidthLogatihmicScale * index);
                        float_64 zq2 = math::pow(2, minZqExponent + stepWidthLogatihmicScale * (index + 1));
                        float_64 f = (zq - zq1) / (zq2 - zq1);

                        /// @todo Test Logarithmic interpolation
                        F1 = F1F2(index, 0);
                        Ftemp = F1F2(index + 1, 0);
                        F1 = math::pow(F1, 1 - f) * math::pow(Ftemp, f); // F1 = F1**((1-f)) * Ftemp**f

                        F2 = F1F2(index, 1);
                        Ftemp = F1F2(index + 1, 1);
                        F2 = math::pow(F2, 1 - f) * math::pow(Ftemp, f); // F2 = F2**((1-f)) * Ftemp**f
                    }

                    //  - Calculating the numeric factor: numericFactor = dt * (e**2 * m_e * c /( hbar**2 * eps0 * 4 *
                    //  np.pi))
                    float_X numericFactor = DELTA_T
                        * (ELECTRON_CHARGE * ELECTRON_CHARGE * ELECTRON_MASS * SPEED_OF_LIGHT
                           / (HBAR * HBAR * EPS0 * 4 * PI));
                    /// @todo change and used unified requirement
                    float_X requirement1 = numericFactor * 1.5 * math::pow(chi, 2. / 3.) / gamma;
                    float_X requirement2 = numericFactor * 0.5 * chi / gamma;
                    if constexpr(T_Debug)
                    {
                        ///@todo maby 0.1 is to small -> check
                        if(requirement1 > 0.1 || requirement2 > 0.1)
                        {
                            printf("Synchrotron Extension requirement1 (should be less than 0.1): %f\n", requirement1);
                            printf("Synchrotron Extension requirement2 (should be less than 0.1): %f\n", requirement2);
                        }
                    }
                    numericFactor *= math::sqrt(3) / (2 * PI);

                    // Calculate propability:
                    ///@todo check the numerical stability -> maby use one numerator
                    float_X const numerator1 = (1 - delta) * chi;
                    float_X const numerator2 = (F1 + 3 * delta * zq * chi / 2 * F2);
                    float_X const denominator = gamma * delta;

                    float_X propability = numericFactor * (numerator1 / denominator * numerator2) * 3 * r1r1;

                    if constexpr(T_Debug)
                        if(propability > randNr2)
                        {
                            printf("propability: %e\n", propability);
                            printf("delta: %e\n", delta);
                            printf("HeffValue: %e\n", HeffValue);
                            printf("chi: %e\n", chi);
                            if(requirement1 > 0.1 || requirement2 > 0.1)
                                printf("\t\t\t\tWARNING\n");
                            printf("requirement1: %e\n", requirement1);
                            printf("requirement2: %e\n", requirement2);
                            printf("Returned: %e\n\n", (propability > randNr2) * delta);
                        }

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

                /* specify field to particle interpolation scheme */
                using Field2ParticleInterpolation = typename pmacc::traits::Resolve<
                    typename pmacc::traits::GetFlagType<FrameType, interpolation<>>::type>::type;

                /* margins around the supercell for the interpolation of the field on the cells */
                using LowerMargin = typename GetMargin<Field2ParticleInterpolation>::LowerMargin;
                using UpperMargin = typename GetMargin<Field2ParticleInterpolation>::UpperMargin;

                /* relevant area of a block */
                using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

                BlockArea BlockDescription;

            private:
                /* random number generator */
                using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                using Distribution = pmacc::random::distributions::Uniform<float_X>;
                using RandomGen = typename RNGFactory::GetRandomType<Distribution>::type;
                RandomGen randomGen;

                using TVec = MappingDesc::SuperCellSize;

                using ValueType_E = FieldE::ValueType;
                using ValueType_B = FieldB::ValueType;
                /* global memory EM-field and current density device databoxes */
                PMACC_ALIGN(eBox, FieldE::DataBoxType);
                PMACC_ALIGN(bBox, FieldB::DataBoxType);
                /* shared memory EM-field device databoxes */
                PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize, 1>>);
                PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize, 0>>);

                // F1F2DeviceBuff_ is a pointer to a databox containing F1 and F2 values. m stands for member
                GridBuffer<float_64, 2>::DataBoxType m_F1F2DeviceBuff;

            public:
                /* host constructor initializing member : random number generator */
                AlgorithmSynchrotron(const uint32_t currentStep, GridBuffer<float_64, 2>::DataBoxType F1F2DeviceBuff)
                    : randomGen(RNGFactory::createRandom<Distribution>())
                {
                    m_F1F2DeviceBuff = F1F2DeviceBuff;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    /* initialize pointers on host-side E-(B-)field and current density databoxes */
                    auto fieldE = dc.get<FieldE>(FieldE::getName());
                    auto fieldB = dc.get<FieldB>(FieldB::getName());
                    /* initialize device-side E-(B-)field and current density databoxes */
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
                    /* caching of E and B fields */
                    cachedB = CachedBox::create<0, ValueType_B>(worker, BlockArea());
                    cachedE = CachedBox::create<1, ValueType_E>(worker, BlockArea());

                    /* instance of nvidia assignment operator */
                    pmacc::math::operation::Assign assign;
                    /* copy fields from global to shared */
                    auto fieldBBlock = bBox.shift(blockCell);
                    auto collective = makeThreadCollective<BlockArea>();
                    collective(worker, assign, cachedB, fieldBBlock);
                    /* copy fields from global to shared */
                    auto fieldEBlock = eBox.shift(blockCell);
                    collective(worker, assign, cachedE, fieldEBlock);

                    /* wait for shared memory to be initialized */
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
                    [[maybe_unused]] T_Worker const& worker,
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
                    /* alias for the single macro-particle - electron */
                    auto particle = electronFrame[localIdx];
                    /* particle position, used for field-to-particle interpolation */
                    floatD_X pos = particle[position_];
                    const int particleCellIdx = particle[localCellIdx_];
                    /* multi-dim coordinate of the local cell inside the super cell */
                    DataSpace<TVec::dim> localCell = pmacc::math::mapToND(TVec::toRT(), particleCellIdx);
                    /* interpolation of E- */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldE> fieldPosE;
                    ValueType_E eField = Field2ParticleInterpolation()(cachedE.shift(localCell), pos, fieldPosE());
                    /*                     and B-field on the particle position */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldB> fieldPosB;
                    ValueType_B bField = Field2ParticleInterpolation()(cachedB.shift(localCell), pos, fieldPosB());

                    // use the algorithm from the SynchrotronIdea struct
                    SynchrotronIdea synchrotronAlgo;
                    float_X photonEnergy = synchrotronAlgo(
                        bField,
                        eField,
                        particle,
                        this->randomGen(worker),
                        this->randomGen(worker),
                        m_F1F2DeviceBuff);

                    ///@todo check if this works in parallel
                    // save to member variable to use in creation of new photon
                    m_normalizedPhotonEnergy = photonEnergy;

                    return (photonEnergy > 0); // generate one photon if energy is > 0
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
                    T_parentElectron& parentElectron,
                    T_childPhoton& childPhoton)
                {
                    /* for not mixing operations::assign up with the nvidia functor assign */
                    namespace partOp = pmacc::particles::operations;
                    /* each thread sets the multiMask hard on "particle" (=1) */
                    childPhoton[multiMask_] = 1u;
                    const float_X weighting = parentElectron[weighting_];

                    /* each thread initializes a clone of the parent Electron but leaving out
                     * some attributes:
                     * - multiMask: reading from global memory takes longer than just setting it again explicitly
                     * - momentum: because the Photon would get a higher energy because of the Electron mass
                     * - boundPhotons: because species other than Electrons or atoms do not have them
                     * (gets AUTOMATICALLY deselected because Photons do not have this attribute)
                     */
                    auto targetPhotonClone = partOp::deselect<pmacc::mp_list<multiMask, momentum>>(childPhoton);

                    partOp::assign(targetPhotonClone, partOp::deselect<particleId>(parentElectron));

                    const float_X massElectron = attribute::getMass(weighting, parentElectron);
                    const float_X massPhoton = attribute::getMass(weighting, childPhoton);

                    ///@todo decide on calculation:
                    // conversion factor from photon energy to momentum
                    constexpr float_X convFactor = 1.0 / SPEED_OF_LIGHT;
                    // if this is wrong uncomment the lines below and comment this line
                    float3_X const PhotonMomentum = parentElectron[momentum_] * m_normalizedPhotonEnergy * convFactor;

                    // float3_X const mom = parentElectron[momentum_];
                    // float_X const mass = attribute::getMass(float_X(1), parentElectron); //weighting 1
                    // float_X const gamma = Gamma<float_X>()(mom, mass);
                    // constexpr float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
                    // float_X const electronEnergy = (gamma - float_X(1.0)) * mass * c2;
                    // float3_X const PhotonMomentum = mom/pmacc::math::l2norm(mom) * photonEnergy * SPEED_OF_LIGHT;

                    childPhoton[momentum_] = PhotonMomentum;


                    if constexpr(T_Debug)
                    {
                        printf("parentElectron[momentum_] y: %e\n", parentElectron[momentum_].y());
                        printf("parentElectron[momentum_] z: %e\n", parentElectron[momentum_].z());
                        printf("PhotonMomentum.x(): %e\n", PhotonMomentum.x());
                        printf("PhotonMomentum.y(): %e\n", PhotonMomentum.y());
                        printf("PhotonMomentum.z(): %e\n", PhotonMomentum.z());
                        printf("m_normalizedPhotonEnergy: %e\n\n\n", m_normalizedPhotonEnergy);
                    }

                    /* conservatElectron of momentum */
                    if constexpr(params::ElectronRecoil)
                        parentElectron[momentum_] -= PhotonMomentum;
                }

            private:
                float_X m_normalizedPhotonEnergy; // energy of emitted photon with respect to electron energy

            }; // end of struct AlgorithmSynchrotron
        } // namespace synchrotron
    } // namespace particles
} // namespace picongpu
