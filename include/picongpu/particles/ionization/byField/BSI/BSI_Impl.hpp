/* Copyright 2015-2023 Marco Garten, Jakob Trojok
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/CellType.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/particles/ParticlesFunctors.hpp"
#include "picongpu/particles/atomicPhysics/SetChargeState.hpp"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSI.hpp"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSIEffectiveZ.hpp"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSIStarkShifted.hpp"
#include "picongpu/particles/ionization/byField/BSI/BSI.def"
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizationCurrent.hpp"
#include "picongpu/traits/FieldPosition.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/meta/conversion/TypeToPointerPair.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/traits/Resolve.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** \struct BSI_Impl
             *
             * \brief Barrier Suppression Ionization - Implementation
             *
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the electron species to be created
             * @tparam T_IonizationCurrent select type of ionization current (None or EnergyConservation)
             * @tparam T_SrcSpecies type or name as PMACC_CSTRING of the particle species that is ionized
             */
            template<
                typename T_IonizationAlgorithm,
                typename T_DestSpecies,
                typename T_IonizationCurrent,
                typename T_SrcSpecies>
            struct BSI_Impl
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
                /* define ionization ALGORITHM (calculation) for ionization MODEL */
                using IonizationAlgorithm = T_IonizationAlgorithm;

                using TVec = MappingDesc::SuperCellSize;

                using ValueType_E = FieldE::ValueType;
                /* global memory E-field and current density device databoxes */
                FieldE::DataBoxType eBox;
                FieldJ::DataBoxType jBox;
                /* shared memory EM-field device databoxes */
                PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize, 1>>);

            public:
                /* host constructor */
                BSI_Impl(const uint32_t currentStep)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    /* initialize pointers on host-side E-field and current density databoxes */
                    auto fieldE = dc.get<FieldE>(FieldE::getName());
                    auto fieldJ = dc.get<FieldJ>(FieldJ::getName());
                    /* initialize device-side E-(J-)field databoxes */
                    eBox = fieldE->getDeviceDataBox();
                    jBox = fieldJ->getDeviceDataBox();
                }

                /** cache fields used by this functor
                 *
                 * @warning this is a collective method and calls synchronize
                 *
                 * @tparam T_Worker lockstep worker type
                 *
                 * @param worker lockstep worker
                 * @param blockCell relative offset (in cells) to the local domain plus the guarding cells
                 * @param worker lockstep worker
                 */
                template<typename T_Worker>
                DINLINE void collectiveInit(const T_Worker& worker, const DataSpace<simDim>& blockCell)
                {
                    /* shift origin of jbox to supercell of particle */
                    jBox = jBox.shift(blockCell);

                    /* caching of E field */
                    cachedE = CachedBox::create<1, ValueType_E>(worker, BlockArea());

                    /* instance of nvidia assignment operator */
                    pmacc::math::operation::Assign assign;

                    auto collective = makeThreadCollective<BlockArea>();
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
                 * during loop execution. The reason for this is the `alpaka::syncBlockThreads( acc )` call which is
                 * necessary after initializing the E-/B-field shared boxes in shared memory.
                 *
                 * @param localSuperCellOffset offset (in superCells, without any guards) relative
                 *                             to the origin of the local domain
                 * @param rngIdx linear index rng number index within the supercell, valid range[0;numFrameSlots)
                 */
                template<typename T_Worker>
                DINLINE void init(
                    [[maybe_unused]] T_Worker const& worker,
                    [[maybe_unused]] const DataSpace<simDim>& localSuperCellOffset,
                    [[maybe_unused]] const uint32_t rngIdx)
                {
                }

                /** Determine number of new macro electrons due to ionization
                 *
                 * @param ionFrame reference to frame of the to-be-ionized particles
                 * @param localIdx local (linear) index in super cell / frame
                 */
                template<typename T_Worker>
                DINLINE uint32_t numNewParticles(T_Worker const& worker, FrameType& ionFrame, int localIdx)
                {
                    /* alias for the single macro-particle */
                    auto particle = ionFrame[localIdx];
                    /* particle position, used for field-to-particle interpolation */
                    floatD_X pos = particle[position_];
                    const int particleCellIdx = particle[localCellIdx_];
                    /* multi-dim coordinate of the local cell inside the super cell */
                    DataSpace<TVec::dim> localCell = pmacc::math::mapToND(TVec::toRT(), particleCellIdx);
                    /* interpolation of E */
                    const picongpu::traits::FieldPosition<fields::CellType, FieldE> fieldPosE;
                    ValueType_E eField = Field2ParticleInterpolation()(cachedE.shift(localCell), pos, fieldPosE());

                    /* this is the point where actual ionization takes place */
                    IonizationAlgorithm ionizeAlgo{};
                    auto retValue = ionizeAlgo(eField, particle);
                    /* determine number of new macro electrons to be created and calculate ionization current */
                    IonizationCurrent<T_DestSpecies, simDim, T_IonizationCurrent>{}(
                        retValue,
                        particle[weighting_],
                        jBox.shift(localCell),
                        eField,
                        worker,
                        pos);

                    return retValue.newMacroElectrons;
                }

                /* Functor implementation
                 *
                 * Ionization model specific particle creation
                 *
                 * @tparam T_parentIon type of ion species that is being ionized
                 * @tparam T_childElectron type of electron species that is created
                 * @param parentIon ion instance that is ionized
                 * @param childElectron electron instance that is created
                 */
                template<typename T_parentIon, typename T_childElectron, typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    IdGenerator& idGen,
                    T_parentIon& parentIon,
                    T_childElectron& childElectron)
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
                    auto targetElectronClone = partOp::deselect<pmacc::mp_list<multiMask, momentum>>(childElectron);

                    targetElectronClone.copyAndInit(worker, idGen, partOp::deselect<particleId>(parentIon));

                    const float_X massIon = attribute::getMass(weighting, parentIon);
                    const float_X massElectron = attribute::getMass(weighting, childElectron);

                    const float3_X electronMomentum(parentIon[momentum_] * (massElectron / massIon));

                    childElectron[momentum_] = electronMomentum;

                    /* conservation of momentum
                     * \todo add conservation of mass */
                    parentIon[momentum_] -= electronMomentum;

                    /** ionization of the ion by reducing the number of bound electrons
                     *
                     * @warning subtracting a float from a float can potentially
                     *          create a negative boundElectrons number for the ion,
                     *          see #1850 for details
                     */
                    float_X numberBoundElectrons = parentIon[boundElectrons_];

                    numberBoundElectrons -= 1._X;

                    picongpu::particles::atomicPhysics::SetChargeState{}(parentIon, numberBoundElectrons);
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
