/**
 * Copyright 2015 Marco Garten
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

#include "types.h"
#include "math/vector/Size_t.hpp"
#include "simulation_defines.hpp"
#include "traits/Resolve.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"

#include "particles/ionization/byField/BSI/BSI.def"
#include "particles/ionization/byField/BSI/AlgorithmBSI.hpp"
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
     * \brief Barrier Suppression Ionization - Implementation
     *
     * \tparam T_DestSpecies electron species to be created
     * \tparam T_SrcSpecies particle species that is ionized
     */
    template<typename T_DestSpecies, typename T_SrcSpecies>
    struct BSI_Impl
    {

        typedef T_DestSpecies DestSpecies;
        typedef T_SrcSpecies  SrcSpecies;

        typedef typename SrcSpecies::FrameType FrameType;

        /* specify field to particle interpolation scheme */
        typedef typename PMacc::traits::Resolve<
            typename GetFlagType<FrameType,interpolation<> >::type
        >::type InterpolationScheme;

        /* margins around the supercell for the interpolation of the field on the cells */
        typedef typename GetMargin<InterpolationScheme>::LowerMargin LowerMargin;
        typedef typename GetMargin<InterpolationScheme>::UpperMargin UpperMargin;

        /* relevant area of a block */
        typedef SuperCellDescription<
            typename MappingDesc::SuperCellSize,
            LowerMargin,
            UpperMargin
            > BlockArea;

        BlockArea BlockDescription;

        private:

            /* define ionization ALGORITHM (calculation) for ionization MODEL */
            typedef particles::ionization::AlgorithmBSI IonizationAlgorithm;

            typedef MappingDesc::SuperCellSize TVec;

            typedef FieldE::ValueType ValueType_E;
            typedef FieldB::ValueType ValueType_B;
            /* global memory EM-field device databoxes */
            FieldE::DataBoxType eBox;
            FieldB::DataBoxType bBox;
            /* shared memory EM-field device databoxes */
            PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize,1> >);
            PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize,0> >);

        public:
            /* host constructor */
            BSI_Impl(const uint32_t currentStep)
            {
                DataConnector &dc = Environment<>::get().DataConnector();
                /* initialize pointers on host-side E-(B-)field databoxes */
                FieldE* fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));
                FieldB* fieldB = &(dc.getData<FieldB > (FieldB::getName(), true));
                /* initialize device-side E-(B-)field databoxes */
                eBox = fieldE->getDeviceDataBox();
                bBox = fieldB->getDeviceDataBox();

            }

            /** init
             *  \brief Cache EM-fields on device
             *         and initialize possible prerequisites for ionization, like e.g. random number generator.
             *
             * This function will be called inline on the device which must happen BEFORE threads diverge
             */
            DINLINE void init(const DataSpace<simDim>& blockCell, const int& linearThreadIdx, const DataSpace<simDim>& totalCellOffset)
            {

                /* caching of E and B fields */
                cachedB = CachedBox::create < 0, ValueType_B > (BlockArea());
                cachedE = CachedBox::create < 1, ValueType_E > (BlockArea());
                /* wait for shared memory to be initialized */
                __syncthreads();
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
            }

            /** Functor implementation
             *
             * \param ionFrame (here address of: ) frame of the to-be-ionized particles
             * \param localIdx local (linear) index in super cell / frame
             * \param newMacroElectrons (here address of: ) variable for each thread that stores the number
             *        of macro electrons to be created during the current time step
             */
            DINLINE void operator()(FrameType& ionFrame, int localIdx, unsigned int& newMacroElectrons)
            {

                /* type for field to particle interpolation */
                typedef InterpolationScheme Field2ParticleInterpolation;

                /* type of PIC-scheme cell */
                typedef typename fieldSolver::NumericalCellType NumericalCellType;

                typedef ValueType_B BType;
                typedef ValueType_E EType;
                /* alias for the single macro-particle */
                PMACC_AUTO(particle,ionFrame[localIdx]);
                /* particle position, used for field-to-particle interpolation */
                floatD_X pos = particle[position_];
                const int particleCellIdx = particle[localCellIdx_];
                /* multi-dim coordinate of the local cell inside the super cell */
                DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));
                /* interpolation of E- */
                EType eField = Field2ParticleInterpolation()
                    (cachedE.shift(localCell).toCursor(), pos, NumericalCellType::getEFieldPosition());
                /*                     and B-field on the particle position */
                BType bField = Field2ParticleInterpolation()
                    (cachedB.shift(localCell).toCursor(), pos, NumericalCellType::getBFieldPosition());

                /* define number of bound macro electrons before ionization */
                float_X prevBoundElectrons = particle[boundElectrons_];

                /* this is the point where actual ionization takes place */
                IonizationAlgorithm ionizeAlgo;
                ionizeAlgo(
                     bField, eField,
                     particle
                     );

                /* determine number of new macro electrons to be created */
                newMacroElectrons = prevBoundElectrons - particle[boundElectrons_];

            }

    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
