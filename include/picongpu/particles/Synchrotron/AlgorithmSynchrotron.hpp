/* Copyright 2014-2023 Marco Garten, Jakob Trojok
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

#include "picongpu/traits/attribute/GetChargeState.hpp"

/** @file AlgorithmSynchrotron.hpp
 *
 * Synchrotron ALGORITHM for the Synchrotron model
 *
 * - implements the calculation of Synchrotron probability and changes charge states
 *   by decreasing the number of bound electrons
 * - is called with the Synchrotron MODEL, specifically by setting the flag in @see speciesDefinition.param
 */

namespace picongpu
{
    namespace particles
    {
        namespace synchrotron
        {
            /** \struct AlgorithmSynchrotron
             *
             * \brief Ammosov-Delone-Krainov
             *        Tunneling ionization for hydrogenlike atoms
             *
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the electron species to be created
             * @tparam T_IonizationCurrent select type of ionization current (None or EnergyConservation)
             * @tparam T_SrcSpecies type or name as PMACC_CSTRING of the particle species that is ionized
             */
            struct SynchrotronIdea
            {
                /** Functor implementation
                 * @tparam EType type of electric field
                 * @tparam BType type of magnetic field
                 * @tparam ParticleType type of particle to be ionized
                 *
                 * @param bField magnetic field value at t=0
                 * @param eField electric field value at t=0
                 * @param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 * @param randNr random number, equally distributed in range [0.:1.0]
                 *
                 * @return ionization energy and number of new macro electrons to be created
                 */

                #define F1F2(zq,i) F1F2DeviceBuff(DataSpace<2>{zq,i})

                template<typename EType, typename BType, typename ParticleType>
                HDINLINE float_X //return type -> energy of a photon relative to the electron energy
                operator()(const BType bField, const EType eField, ParticleType& parentIon, float_X randNr1, float_X randNr2, GridBuffer<float_64,2>::DataBoxType F1F2DeviceBuff) const
                {
                    // print something from  F1F2
                    // for (uint32_t zq = 500; zq < 999; zq++) {
                        // std::cout << "zq = " << zq << " F1 = " << F1F2DeviceBuff(DataSpace<2>{zq,0}) << " F2 = " << F1F2DeviceBuff(DataSpace<2>{zq,1}) << std::endl;
                        // std::cout << "zq = " << zq << " F1 = " << F1F2(zq,0) << " F2 = " << F1F2(zq,1) << std::endl;
                    // }

                    //Check the interpolation
                    int16_t minZqExp = -18;
                    int16_t maxZqExp = 1;
                    float_64 zq = (randNr1+randNr2)*5;
                    if (zq){
                        int16_t index = (std::log10(zq) - minZqExp) / (maxZqExp - minZqExp) * 1000;
                        float_64 F1 = F1F2(index,0);
                        float_64 F2 = F1F2(index,1);
                        printf("zq: %f\n", zq);
                        printf("index: %d\n", index);
                        printf("F1: %f\n", F1);
                        printf("F2: %f\n", F2);
                    }


                    // printf random numbers -> they are random indeed
                    printf("randNr1: %f\n", randNr1);
                    printf("randNr2: %f\n", randNr2);

                    // calculate Heff

                    // Calculate chi and z_q
                    // float_64 chi = e * hbar * HeffValue * gamma / (m_e * m_e * c * c * c);
                    // float_64 z_q = 2 * delta / (3 * chi * (1 - delta)); // Assuming delta is defined

                    // calculate propability
                    // calculate energy of photon
                    
                    
                    return float_X(1.);
                }
            };

            template<
                typename T_SrcSpecies,
                typename T_DestSpecies>
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
                
                // F1F2DeviceBuff_ is a pointer to a databox containing F1 and F2 values
                GridBuffer<float_64,2>::DataBoxType F1F2DeviceBuff_;

            public:
                /* host constructor initializing member : random number generator */
                AlgorithmSynchrotron(const uint32_t currentStep, GridBuffer<float_64,2>::DataBoxType F1F2DeviceBuff) : randomGen(RNGFactory::createRandom<Distribution>())
                {
                    F1F2DeviceBuff_ = F1F2DeviceBuff;
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
                 *         and initialize possible prerequisites for ionization, like e.g. random number generator.
                 *
                 * This function will be called inline on the device which must happen BEFORE threads diverge
                 * during loop execution. The reason for this is the `cupla::__syncthreads( acc )` call which is
                 * necessary after initializing the E-/B-field shared boxes in shared memory.
                 */
                template<typename T_Worker>
                DINLINE void init(
                    T_Worker const& worker,
                    const DataSpace<simDim>& blockCell,
                    const DataSpace<simDim>& localCellOffset)
                {
                    /* initialize random number generator with the local cell index in the simulation */
                    this->randomGen.init(localCellOffset);
                }

                /** Determine number of new macro electrons due to ionization
                 *
                 * @param ionFrame reference to frame of the to-be-ionized particles
                 * @param localIdx local (linear) index in super cell / frame
                 */
                template<typename T_Worker>
                DINLINE uint32_t numNewParticles(const T_Worker& worker, FrameType& ionFrame, int localIdx)
                {
                    /* alias for the single macro-particle */
                    auto particle = ionFrame[localIdx];
                    /* particle position, used for field-to-particle interpolation */
                    floatD_X pos = particle[position_];
                    const int particleCellIdx = particle[localCellIdx_];
                    /* multi-dim coordinate of the local cell inside the super cell */
                    DataSpace<TVec::dim> localCell(
                        DataSpaceOperations<TVec::dim>::template map<TVec>(particleCellIdx));
                    /* interpolation of E- */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldE> fieldPosE;
                    ValueType_E eField = Field2ParticleInterpolation()(cachedE.shift(localCell), pos, fieldPosE());
                    /*                     and B-field on the particle position */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldB> fieldPosB;
                    ValueType_B bField = Field2ParticleInterpolation()(cachedB.shift(localCell), pos, fieldPosB());

                    SynchrotronIdea synchrotronAlgo;
                    printf("worker: %u\n", worker.getWorkerIdx());
                    printf("cpu rand: %f\n", this->randomGen(worker));

                    float_X photonEnergy = synchrotronAlgo(bField, eField, particle, this->randomGen(worker), this->randomGen(worker), F1F2DeviceBuff_); // can I do 2x random number -> it looks like yes
                    
                    normalizedPhotonEnergy = photonEnergy;
                    return (photonEnergy > 0); // generate photon if energy is > 0 
                }

                /* Functor implementation
                 *
                 * Ionization model specific particle creation
                 *
                 * @tparam T_parentIon type of ion species that is being ionized
                 * @tparam T_childPhoton type of Photon species that is created
                 * @param parentIon ion instance that is ionized
                 * @param childPhoton Photon instance that is created
                 */
                template<typename T_parentElectron, typename T_childPhoton, typename T_Worker>
                DINLINE void operator()(const T_Worker& worker, T_parentElectron& parentElectron, T_childPhoton& childPhoton)
                {
                    /* for not mixing operatElectrons::assign up with the nvidia functor assign */
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

                    // const float_X energyPhoton = childPhoton[energy_];
                    double convFactor = 1.0;
                    const float3_X PhotonMomentum(parentElectron[momentum_] * normalizedPhotonEnergy * convFactor );

                    childPhoton[momentum_] = PhotonMomentum;

                    /* conservatElectron of momentum
                     * \todo add conservatElectron of mass */
                    parentElectron[momentum_] -= PhotonMomentum;

                    /** ElectronizatElectron of the Electron by reducing the number of bound Photons
                     *
                     * @warning subtracting a float from a float can potentially
                     *          create a negative boundPhotons number for the Electron,
                     *          see #1850 for details
                     */
                    // float_X numberBoundPhotons = parentElectron[boundPhotons_];

                    // numberBoundPhotons -= 1._X;

                    // picongpu::particles::atomicPhysics::SetChargeState{}(parentElectron, numberBoundPhotons);
                }
                private:
                float_X normalizedPhotonEnergy; // energy of a emitted photon with respect to electron

            };// end of struct AlgorithmSynchrotron
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
