/* Copyright 2015-2019 Marco Garten
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

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/particles/ionization/byField/BSI/BSI.def"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSI.hpp"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSIEffectiveZ.hpp"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSIStarkShifted.hpp"

#include "picongpu/particles/ParticlesFunctors.hpp"

#include <pmacc/compileTime/conversion/TypeToPointerPair.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/particles/compileTime/FindByNameOrType.hpp>
#include <pmacc/mappings/threads/WorkerCfg.hpp>


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
     * \tparam T_DestSpecies type or name as boost::mpl::string of the electron species to be created
     * \tparam T_SrcSpecies type or name as boost::mpl::string of the particle species that is ionized
     */
    template<typename T_IonizationAlgorithm, typename T_DestSpecies, typename T_SrcSpecies>
    struct BSI_Impl
    {

        using DestSpecies = pmacc::particles::compileTime::FindByNameOrType_t<
            VectorAllSpecies,
            T_DestSpecies
        >;
        using SrcSpecies = pmacc::particles::compileTime::FindByNameOrType_t<
            VectorAllSpecies,
            T_SrcSpecies
        >;

        using FrameType = typename SrcSpecies::FrameType;

        /* specify field to particle interpolation scheme */
        using Field2ParticleInterpolation = typename pmacc::traits::Resolve<
            typename GetFlagType<FrameType,interpolation<> >::type
        >::type;

        /* margins around the supercell for the interpolation of the field on the cells */
        using LowerMargin = typename GetMargin<Field2ParticleInterpolation>::LowerMargin;
        using UpperMargin = typename GetMargin<Field2ParticleInterpolation>::UpperMargin;

        /* relevant area of a block */
        using BlockArea = SuperCellDescription<
            typename MappingDesc::SuperCellSize,
            LowerMargin,
            UpperMargin
        >;

        BlockArea BlockDescription;

        private:

            /* define ionization ALGORITHM (calculation) for ionization MODEL */
            using IonizationAlgorithm = T_IonizationAlgorithm;

            using TVec = MappingDesc::SuperCellSize;

            using ValueType_E = FieldE::ValueType;
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
                auto fieldE = dc.get< FieldE >( FieldE::getName(), true );
                /* initialize device-side E-(B-)field databoxes */
                eBox = fieldE->getDeviceDataBox();

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
            template<
                typename T_Acc ,
                typename T_WorkerCfg
            >
            DINLINE void collectiveInit(
                const T_Acc & acc,
                const DataSpace<simDim>& blockCell,
                const T_WorkerCfg & workerCfg
            )
            {

                /* caching of E field */
                cachedE = CachedBox::create<
                    1,
                    ValueType_E
                >(
                    acc,
                    BlockArea()
                );

                /* instance of nvidia assignment operator */
                nvidia::functors::Assign assign;

                ThreadCollective<
                    BlockArea,
                    T_WorkerCfg::numWorkers
                > collective( workerCfg.getWorkerIdx( ) );
                /* copy fields from global to shared */
                auto fieldEBlock = eBox.shift(blockCell);
                collective(
                          acc,
                          assign,
                          cachedE,
                          fieldEBlock
                          );

                /* wait for shared memory to be initialized */
                __syncthreads();
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
            template< typename T_Acc >
            DINLINE void init(
                T_Acc const & acc,
                const DataSpace<simDim>& blockCell,
                const int& linearThreadIdx,
                const DataSpace<simDim>& localCellOffset
            )
            {
            }

            /** Determine number of new macro electrons due to ionization
             *
             * \param ionFrame reference to frame of the to-be-ionized particles
             * \param localIdx local (linear) index in super cell / frame
             */
            template< typename T_Acc >
            DINLINE uint32_t numNewParticles(T_Acc const & acc, FrameType& ionFrame, int localIdx)
            {
                /* alias for the single macro-particle */
                auto particle = ionFrame[localIdx];
                /* particle position, used for field-to-particle interpolation */
                floatD_X pos = particle[position_];
                const int particleCellIdx = particle[localCellIdx_];
                /* multi-dim coordinate of the local cell inside the super cell */
                DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));
                /* interpolation of E */
                const picongpu::traits::FieldPosition<typename fields::Solver::NummericalCellType, FieldE> fieldPosE;
                ValueType_E eField = Field2ParticleInterpolation()
                    (cachedE.shift(localCell).toCursor(), pos, fieldPosE());

                /* define number of bound macro electrons before ionization */
                float_X prevBoundElectrons = particle[boundElectrons_];

                /* this is the point where actual ionization takes place */
                IonizationAlgorithm ionizeAlgo;
                /* determine number of new macro electrons to be created */
                uint32_t newMacroElectrons = ionizeAlgo(
                                                eField,
                                                particle
                                              );

                return newMacroElectrons;

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
            DINLINE void operator()(T_Acc const & acc, T_parentIon& parentIon,T_childElectron& childElectron)
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
                auto targetElectronClone = partOp::deselect<bmpl::vector2<multiMask, momentum> >(childElectron);

                partOp::assign(targetElectronClone, partOp::deselect<particleId>(parentIon));

                const float_X massIon = attribute::getMass(weighting,parentIon);
                const float_X massElectron = attribute::getMass(weighting,childElectron);

                const float3_X electronMomentum (parentIon[momentum_]*(massElectron/massIon));

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
