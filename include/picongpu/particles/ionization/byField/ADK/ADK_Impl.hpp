/* Copyright 2015-2021 Marco Garten, Jakob Trojok
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
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include "picongpu/fields/CellType.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/particles/ionization/byField/ADK/ADK.def"
#include "picongpu/particles/ionization/byField/ADK/AlgorithmADK.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/JIonizationCalc.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/JIonizationAssignment.hpp"

#include <pmacc/random/methods/methods.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/conversion/TypeToPointerPair.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/threads/WorkerCfg.hpp>

#include <boost/type_traits/integral_constant.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** \struct ADK_Impl
             *
             * \brief Ammosov-Delone-Krainov
             *        Tunneling ionization for hydrogenlike atoms
             *
             * \tparam T_DestSpecies type or name as boost::mpl::string of the electron species to be created
             * \tparam T_IonizationCurrent select type of ionization current (None or EnergyConservation)
             * \tparam T_SrcSpecies type or name as boost::mpl::string of the particle species that is ionized
             */
            template<
                typename T_IonizationAlgorithm,
                typename T_DestSpecies,
                typename T_IonizationCurrent,
                typename T_SrcSpecies>
            struct ADK_Impl
            {
                using DestSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_DestSpecies>;
                using SrcSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SrcSpecies>;

                using FrameType = typename SrcSpecies::FrameType;

                /* specify field to particle interpolation scheme */
                using Field2ParticleInterpolation =
                    typename pmacc::traits::Resolve<typename GetFlagType<FrameType, interpolation<>>::type>::type;

                /* margins around the supercell for the interpolation of the field on the cells */
                using LowerMargin = typename GetMargin<Field2ParticleInterpolation>::LowerMargin;
                using UpperMargin = typename GetMargin<Field2ParticleInterpolation>::UpperMargin;

                /* relevant area of a block */
                using BlockArea = SuperCellDescription<typename MappingDesc::SuperCellSize, LowerMargin, UpperMargin>;

                BlockArea BlockDescription;

            private:
                /* define ionization ALGORITHM (calculation) for ionization MODEL */
                using IonizationAlgorithm = T_IonizationAlgorithm;

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
                PMACC_ALIGN(jBox, FieldJ::DataBoxType);
                /* shared memory EM-field device databoxes */
                PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize, 1>>);
                PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize, 0>>);

            public:
                /* host constructor initializing member : random number generator */
                ADK_Impl(const uint32_t currentStep) : randomGen(RNGFactory::createRandom<Distribution>())
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    /* initialize pointers on host-side E-(B-)field and current density databoxes */
                    auto fieldE = dc.get<FieldE>(FieldE::getName(), true);
                    auto fieldB = dc.get<FieldB>(FieldB::getName(), true);
                    auto fieldJ = dc.get<FieldJ>(FieldJ::getName(), true);
                    /* initialize device-side E-(B-)field and current density databoxes */
                    eBox = fieldE->getDeviceDataBox();
                    bBox = fieldB->getDeviceDataBox();
                    jBox = fieldJ->getDeviceDataBox();
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
                    /* shift origin of jbox to supercell of particle */
                    jBox = jBox.shift(blockCell);

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

                /** Determine number of new macro electrons due to ionization
                 *
                 * \param ionFrame reference to frame of the to-be-ionized particles
                 * \param localIdx local (linear) index in super cell / frame
                 */
                template<typename T_Acc>
                DINLINE uint32_t numNewParticles(const T_Acc& acc, FrameType& ionFrame, int localIdx)
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
                    ValueType_E eField
                        = Field2ParticleInterpolation()(cachedE.shift(localCell).toCursor(), pos, fieldPosE());
                    /*                     and B-field on the particle position */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldB> fieldPosB;
                    ValueType_B bField
                        = Field2ParticleInterpolation()(cachedB.shift(localCell).toCursor(), pos, fieldPosB());

                    /* define number of bound macro electrons before ionization */
                    float_X prevBoundElectrons = particle[boundElectrons_];

                    IonizationAlgorithm ionizeAlgo;
                    /* determine number of new macro electrons to be created and energy used for ionization */
                    auto retValue = ionizeAlgo(bField, eField, particle, this->randomGen(acc));
                    IonizationCurrent<T_Acc, T_DestSpecies, simDim, T_IonizationCurrent>{}(
                        retValue,
                        particle[weighting_],
                        jBox.shift(localCell),
                        eField,
                        acc,
                        pos);

                    return retValue.newMacroElectrons;
                }

                /* Functor implementation
                 *
                 * Ionization model specific particle creation
                 *
                 * \tparam T_parentIon type of ion species that is being ionized
                 * \tparam T_childElectron type of electron species that is created
                 * \param parentIon ion instance that is ionized
                 * \param childElectron electron instance that is created
                 */
                template<typename T_parentIon, typename T_childElectron, typename T_Acc>
                DINLINE void operator()(const T_Acc& acc, T_parentIon& parentIon, T_childElectron& childElectron)
                {
                    /* for not mixing operations::assign up with the nvidia functor assign */
                    namespace partOp = pmacc::particles::operations;
                    /* each thread sets the multiMask hard on "particle" (=1) */
                    childElectron[multiMask_] = 1u;
                    const float_X weighting = parentIon[weighting_];

                    /* each thread initializes a clone of the parent ion but leaving out
                     * some attributes:
                     * - multiMask: reading from global memory takes longer than just setting it again explicitly
                     * - momentum: because the electron would get a higher energy because of the ion mass
                     * - boundElectrons: because species other than ions or atoms do not have them
                     * (gets AUTOMATICALLY deselected because electrons do not have this attribute)
                     */
                    auto targetElectronClone = partOp::deselect<bmpl::vector2<multiMask, momentum>>(childElectron);

                    partOp::assign(targetElectronClone, partOp::deselect<particleId>(parentIon));

                    const float_X massIon = attribute::getMass(weighting, parentIon);
                    const float_X massElectron = attribute::getMass(weighting, childElectron);

                    const float3_X electronMomentum(parentIon[momentum_] * (massElectron / massIon));

                    childElectron[momentum_] = electronMomentum;

                    /* conservation of momentum
                     * \todo add conservation of mass */
                    parentIon[momentum_] -= electronMomentum;

                    /** ionization of the ion by reducing the number of bound electrons
                     *
                     * @warning substracting a float from a float can potentially
                     *          create a negative boundElectrons number for the ion,
                     *          see #1850 for details
                     */
                    parentIon[boundElectrons_] -= float_X(1.);
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
