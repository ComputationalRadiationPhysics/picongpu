/**
 * Copyright 2015-2017 Marco Garten
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
#include "traits/Resolve.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "particles/ionization/byField/BSI/BSI.def"
#include "particles/ionization/byField/BSI/AlgorithmBSIHydrogenLike.hpp"
#include "particles/ionization/byField/BSI/AlgorithmBSIEffectiveZ.hpp"
#include "particles/ionization/byField/BSI/AlgorithmBSIStarkShifted.hpp"
#include "particles/ionization/ionization.hpp"

#include "compileTime/conversion/TypeToPointerPair.hpp"
#include "memory/boxes/DataBox.hpp"

#include "particles/ParticlesFunctors.hpp"

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
     * \tparam T_DestSpecies electron species to be created
     * \tparam T_SrcSpecies particle species that is ionized
     */
    template<typename T_IonizationAlgorithm, typename T_DestSpecies, typename T_SrcSpecies>
    struct BSI_Impl
    {

        typedef T_DestSpecies DestSpecies;
        typedef T_SrcSpecies  SrcSpecies;

        typedef typename SrcSpecies::FrameType FrameType;

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

        private:

            /* define ionization ALGORITHM (calculation) for ionization MODEL */
            typedef T_IonizationAlgorithm IonizationAlgorithm;

            typedef MappingDesc::SuperCellSize TVec;

            typedef FieldE::ValueType ValueType_E;
            /* global memory EM-field device databoxes */
            FieldE::DataBoxType eBox;
            /* shared memory EM-field device databoxes */
            PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize,1> >);

        public:
            /* host constructor */
            BSI_Impl(const uint32_t currentStep)
            {
                DataConnector &dc = Environment<>::get().DataConnector();
                /* initialize pointers on host-side E-(B-)field databoxes */
                FieldE* fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));
                /* initialize device-side E-(B-)field databoxes */
                eBox = fieldE->getDeviceDataBox();

            }

            /** Initialization function on device
             *
             * \brief Cache EM-fields on device
             *         and initialize possible prerequisites for ionization, like e.g. random number generator.
             *
             * This function will be called inline on the device which must happen BEFORE threads diverge
             * during loop execution. The reason for this is the `__syncthreads()` call which is necessary after
             * initializing the E-/B-field shared boxes in shared memory.
             *
             * @param blockCell Offset of the cell from the origin of the local domain
             *                  <b>including guarding supercells</b> in units of cells
             * @param linearThreadIdx Linearized thread ID inside the block
             * @param localCellOffset Offset of the cell from the origin of the local
             *                        domain, i.e. from the @see BORDER
             *                        <b>without guarding supercells</b>
             */
            DINLINE void init(const DataSpace<simDim>& blockCell, const int& linearThreadIdx, const DataSpace<simDim>& localCellOffset)
            {

                /* caching of E field */
                cachedE = CachedBox::create < 1, ValueType_E > (BlockArea());

                /* instance of nvidia assignment operator */
                nvidia::functors::Assign assign;

                ThreadCollective<BlockArea> collective(linearThreadIdx);
                /* copy fields from global to shared */
                auto fieldEBlock = eBox.shift(blockCell);
                collective(
                          assign,
                          cachedE,
                          fieldEBlock
                          );
            }

            /** Functor implementation
             *
             * \param ionFrame reference to frame of the to-be-ionized particles
             * \param localIdx local (linear) index in super cell / frame
             * \param newMacroElectrons reference to variable for each thread that stores the number
             *        of macro electrons to be created during the current time step
             */
            DINLINE void operator()(FrameType& ionFrame, int localIdx, unsigned int& newMacroElectrons)
            {
                /* alias for the single macro-particle */
                auto particle = ionFrame[localIdx];
                /* particle position, used for field-to-particle interpolation */
                floatD_X pos = particle[position_];
                const int particleCellIdx = particle[localCellIdx_];
                /* multi-dim coordinate of the local cell inside the super cell */
                DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));
                /* interpolation of E */
                const fieldSolver::numericalCellType::traits::FieldPosition<FieldE> fieldPosE;
                ValueType_E eField = Field2ParticleInterpolation()
                    (cachedE.shift(localCell).toCursor(), pos, fieldPosE());

                /* define number of bound macro electrons before ionization */
                float_X prevBoundElectrons = particle[boundElectrons_];

                /* this is the point where actual ionization takes place */
                IonizationAlgorithm ionizeAlgo;
                /* determine number of new macro electrons to be created */
                newMacroElectrons = ionizeAlgo(
                                                eField,
                                                particle
                                              );

                /* relocate this to writeElectronIntoFrame in ionizationMethods.hpp */
                particle[boundElectrons_] -= float_X(newMacroElectrons);

            }

    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
