/* Copyright 2016-2017 Marco Garten, Axel Huebl
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
#include "traits/UsesRNG.hpp"

#include "fields/FieldTmp.hpp"

#include "particles/ionization/byCollision/ThomasFermi/ThomasFermi.def"
#include "particles/ionization/byCollision/ThomasFermi/AlgorithmThomasFermi.hpp"
#include "particles/ionization/ionization.hpp"
#include "particles/ionization/ionizationMethods.hpp"

#include "random/methods/XorMin.hpp"
#include "random/distributions/Uniform.hpp"
#include "random/RNGProvider.hpp"
#include "dataManagement/DataConnector.hpp"
#include "compileTime/conversion/TypeToPointerPair.hpp"
#include "memory/boxes/DataBox.hpp"
#include "mappings/kernel/AreaMapping.hpp"

#include <boost/type_traits/integral_constant.hpp>

namespace picongpu
{
namespace traits
{
    /** specialization of the UsesRNG trait
     * --> ionization module uses random number generation
     */
    template<typename T_IonizationAlgorithm, typename T_DestSpecies, typename T_SrcSpecies>
    struct UsesRNG<particles::ionization::ThomasFermi_Impl<T_IonizationAlgorithm, T_DestSpecies, T_SrcSpecies> > :
    public boost::true_type
    {
    };
} // namespace traits

namespace particles
{
namespace ionization
{

    /** ThomasFermi_Impl
     *
     * Thomas-Fermi pressure ionization - Implementation
     *
     * @tparam T_IonizationAlgorithm functor that returns a number of
     *         new free macro electrons to create, range: [0, boundElectrons]
     * @tparam T_DestSpecies electron species to be created
     * @tparam T_SrcSpecies particle species that is ionized
     */
    template<typename T_IonizationAlgorithm, typename T_DestSpecies, typename T_SrcSpecies>
    struct ThomasFermi_Impl
    {

        using DestSpecies = T_DestSpecies;
        using SrcSpecies = T_SrcSpecies;

        using FrameType =  typename SrcSpecies::FrameType;

        /** specify field to particle interpolation scheme
         *
         * @todo this needs to be done independently/twice if ion species (rho) and electron
         *       species (ene) are of different shape
         */
        using Field2ParticleInterpolation = typename PMacc::traits::Resolve<
            typename GetFlagType<FrameType,interpolation<> >::type
        >::type;

        /* margins around the supercell for the interpolation of the field on the cells */
        using LowerMargin = typename GetMargin<Field2ParticleInterpolation>::LowerMargin ;
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
            using IonizationAlgorithm =  T_IonizationAlgorithm;

            /* random number generator */
            using RNGFactory = PMacc::random::RNGProvider<simDim, PMacc::random::methods::XorMin>;
            using Distribution = PMacc::random::distributions::Uniform<float>;
            using RandomGen = typename RNGFactory::GetRandomType<Distribution>::type;
            RandomGen randomGen;

            using SuperCellSize = MappingDesc::SuperCellSize;

            using ValueType_Rho = FieldTmp::ValueType;
            using ValueType_Ene = FieldTmp::ValueType ;

            /* global memory EM-field device databoxes */
            PMACC_ALIGN(rhoBox, FieldTmp::DataBoxType);
            PMACC_ALIGN(eneBox, FieldTmp::DataBoxType);

            /* shared memory EM-field device databoxes */
            PMACC_ALIGN(cachedRho, DataBox<SharedBox<ValueType_Rho, typename BlockArea::FullSuperCellSize,0> >);
            PMACC_ALIGN(cachedEne, DataBox<SharedBox<ValueType_Ene, typename BlockArea::FullSuperCellSize,1> >);

            /** Solver for density of the ion species
             *
             *  @todo Include all ion species because the model requires the
             *        density of ionic potential wells
             */
            using DensitySolver = typename particleToGrid::CreateDensityOperation<T_SrcSpecies>::type::Solver;

            /** Solver for energy density of the electron species
             *
             *  @todo Include all electron species with a ForEach<VectorallSpecies,...>
             * instead of just the destination species
             */
            using EnergyDensitySolver = typename particleToGrid::CreateEnergyDensityOperation<T_DestSpecies>::type::Solver;



        public:
            /* host constructor initializing member : random number generator */
            ThomasFermi_Impl(const uint32_t currentStep) : randomGen(RNGFactory::createRandom<Distribution>())
            {
                /* create handle for access to host and device data */
                DataConnector &dc = Environment<>::get().DataConnector();

                /* The compiler is allowed to evaluate an expression that does not depend on a template parameter
                 * even if the class is never instantiated. In that case static assert is always
                 * evaluated (e.g. with clang), this results in an error if the condition is false.
                 * http://www.boost.org/doc/libs/1_60_0/doc/html/boost_staticassert.html
                 *
                 * A workaround is to add a template dependency to the expression.
                 * `sizeof(ANY_TYPE) != 0` is always true and defers the evaluation.
                 */
                PMACC_CASSERT_MSG(
                    _please_allocate_at_least_two_FieldTmp_slots_in_memory_param,
                    ( fieldTmpNumSlots >= 2 ) && ( sizeof( T_IonizationAlgorithm ) != 0 )
                );
                /* initialize pointers on host-side density-/energy density field databoxes */
                auto density = dc.get< FieldTmp >( FieldTmp::getUniqueId( 0 ), true );
                auto eneKinDens = dc.get< FieldTmp >( FieldTmp::getUniqueId( 1 ), true );

                /* reset density and kinetic energy values to zero */
                density->getGridBuffer().getDeviceBuffer().setValue( FieldTmp::ValueType( 0. ) );
                eneKinDens->getGridBuffer().getDeviceBuffer().setValue( FieldTmp::ValueType( 0. ) );

                /* load species without copying the particle data to the host */
                auto srcSpecies = dc.get< SrcSpecies >( SrcSpecies::FrameType::getName(), true );

                /* kernel call for weighted ion density calculation */
                density->template computeValue< CORE + BORDER, DensitySolver >(*srcSpecies, currentStep);
                dc.releaseData( SrcSpecies::FrameType::getName() );
                EventTask densityEvent = density->asyncCommunication( __getTransactionEvent() );
                densityEvent += density->asyncCommunicationGather( densityEvent );

                /* load species without copying the particle data to the host */
                auto destSpecies = dc.get< DestSpecies >( DestSpecies::FrameType::getName(), true );

                /* kernel call for weighted electron energy density calculation */
                eneKinDens->template computeValue< CORE + BORDER, EnergyDensitySolver >(*destSpecies, currentStep);
                dc.releaseData( DestSpecies::FrameType::getName() );
                EventTask eneKinEvent = eneKinDens->asyncCommunication( __getTransactionEvent() );
                eneKinEvent += eneKinDens->asyncCommunicationGather( eneKinEvent );

                /* contributions from neighboring GPUs to our border area */
                __setTransactionEvent( densityEvent + eneKinEvent );

                /* initialize device-side density- and energy density field databox pointers */
                rhoBox = density->getDeviceDataBox();
                eneBox = eneKinDens->getDeviceDataBox();

            }

            /** Initialization function on device
             *
             * Cache density and energy density fields on device and initialize
             * possible prerequisites for ionization, like e.g. random number
             * generator.
             *
             * This function will be called inline on the device which must happen BEFORE threads diverge
             * during loop execution. The reason for this is the `__syncthreads()` call which is necessary after
             * initializing the field shared boxes in shared memory.
             *
             * @param blockCell Offset of the cell from the origin of the local domain
             *                  *including guarding supercells* in units of cells
             * @param linearThreadIdx Linearized thread ID inside the block
             * @param localCellOffset Offset of the cell from the origin of the local
             *                        domain, i.e. from the @see BORDER
             *                        *without guarding supercells*
             */
            DINLINE void init(const DataSpace<simDim>& blockCell, const int& linearThreadIdx, const DataSpace<simDim>& localCellOffset)
            {

                /* caching of density and "temperature" fields */
                cachedRho = CachedBox::create < 0, ValueType_Rho > (BlockArea());
                cachedEne = CachedBox::create < 1, ValueType_Ene > (BlockArea());

                /* instance of nvidia assignment operator */
                nvidia::functors::Assign assign;
                /* copy fields from global to shared */
                auto fieldRhoBlock = rhoBox.shift(blockCell);
                ThreadCollective<BlockArea> collective(linearThreadIdx);
                collective(
                          assign,
                          cachedRho,
                          fieldRhoBlock
                          );
                /* copy fields from global to shared */
                auto fieldEneBlock = eneBox.shift(blockCell);
                collective(
                          assign,
                          cachedEne,
                          fieldEneBlock
                          );

                /* wait for shared memory to be initialized */
                __syncthreads();

                /* initialize random number generator with the local cell index in the simulation */
                this->randomGen.init(localCellOffset);
            }

            /** Functor implementation
             *
             * @param ionFrame reference to frame of the to-be-ionized particles
             * @param localIdx local (linear) index in super cell / frame
             * @param numNewFreeMacroElectrons reference to variable for each
             *        thread that stores the number of macro electrons to be
             *        created during the current time step
             */
            DINLINE void operator()(FrameType& ionFrame, int localIdx, unsigned int& numNewFreeMacroElectrons)
            {
                /* alias for the single macro-particle */
                auto particle = ionFrame[localIdx];
                /* particle position, used for field-to-particle interpolation */
                floatD_X const pos = particle[position_];
                int const particleCellIdx = particle[localCellIdx_];
                /* multi-dim coordinate of the local cell inside the super cell */
                DataSpace<SuperCellSize::dim> localCell(DataSpaceOperations<SuperCellSize::dim>::template map<SuperCellSize > (particleCellIdx));
                /* interpolation of density */
                const fieldSolver::numericalCellType::traits::FieldPosition<FieldTmp> fieldPosRho;
                ValueType_Rho densityV = Field2ParticleInterpolation()
                    (cachedRho.shift(localCell).toCursor(), pos, fieldPosRho());
                /*                          and energy density field on the particle position */
                const fieldSolver::numericalCellType::traits::FieldPosition<FieldTmp> fieldPosEne;
                ValueType_Ene kinEnergyV = Field2ParticleInterpolation()
                    (cachedEne.shift(localCell).toCursor(), pos, fieldPosEne());

                /* density in sim units */
                float_X const density = densityV[0];
                /* energy density in sim units */
                float_X const kinEnergyDensity = kinEnergyV[0];

                /* Returns the new number of bound electrons for an integer number of macro electrons */
                IonizationAlgorithm ionizeAlgo;
                numNewFreeMacroElectrons = ionizeAlgo(
                    kinEnergyDensity,
                    density,
                    particle,
                    this->randomGen()
                );

                /** ionization of the ion by reducing the number of bound electrons
                 *
                 * optimization: only accesses global memory if the charge
                 *               state did really change
                 *
                 * @warning substracting a float from a float can potentially
                 *          create a negative boundElectrons number for the ion,
                 *          see #1850 for details
                 */
                if( numNewFreeMacroElectrons > 0u )
                    particle[ boundElectrons_ ] -= float_X( numNewFreeMacroElectrons );
            }

    };

} // namespace ionization
} // namespace particles
} // namespace picongpu
