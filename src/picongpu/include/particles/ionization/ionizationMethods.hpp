/**
 * Copyright 2014-2015 Marco Garten
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

/** \file
 * This file contains methods needed for ionization like: particle creation functors  */

#pragma once

#include "types.h"
#include "particles/operations/Assign.hpp"
#include "traits/attribute/GetMass.hpp"

#include "nvidia/rng/RNG.hpp"
#include "nvidia/rng/methods/Xor.hpp"
#include "nvidia/rng/distributions/Uniform_float.hpp"
#include "mpi/SeedPerRank.hpp"
#include "traits/GetUniqueTypeId.hpp"

namespace picongpu
{
namespace particles
{
namespace ionization
{

    using namespace PMacc;

    /* operations on particles */
    namespace partOp = PMacc::particles::operations;

    /** \struct WriteElectronIntoFrame
     *
     * \brief functor that fills an electron frame entry with details about the created particle
     */
    struct WriteElectronIntoFrame
    {
        /** Functor implementation
         *
         * \tparam T_parentIon type of the particle which is ionized
         * \tparam T_childElectron type of the electron that will be created
         */
        template<typename T_parentIon, typename T_childElectron>
        DINLINE void operator()(T_parentIon& parentIon,T_childElectron& childElectron)
        {

            /* each thread sets the multiMask hard on "particle" (=1) */
            childElectron[multiMask_] = 1;
            const uint32_t weighting = parentIon[weighting_];

            /* each thread initializes a clone of the parent ion but leaving out
             * some attributes:
             * - multiMask: reading from global memory takes longer than just setting it again explicitly
             * - momentum: because the electron would get a higher energy because of the ion mass
             * - boundElectrons: because species other than ions or atoms do not have them
             * (gets AUTOMATICALLY deselected because electrons do not have this attribute)
             */
            PMACC_AUTO(targetElectronClone, partOp::deselect<bmpl::vector2<multiMask, momentum> >(childElectron));

            partOp::assign(targetElectronClone, parentIon);

            float_X massIon = attribute::getMass(weighting,parentIon);
            const float_X massElectron = attribute::getMass(weighting,childElectron);

            float3_X electronMomentum (parentIon[momentum_]*(massElectron/massIon));

            childElectron[momentum_] = electronMomentum;

            /* conservation of momentum
             * \todo add conservation of mass */
            parentIon[momentum_] -= electronMomentum;
        }
    };

    /* Random number generation */
    namespace nvrng = nvidia::rng;
    namespace rngMethods = nvidia::rng::methods;
    namespace rngDistributions = nvidia::rng::distributions;

    template<typename T_SpeciesType>
    struct RandomNrForMonteCarlo
    {
        typedef T_SpeciesType SpeciesType;
        typedef typename MakeIdentifier<SpeciesType>::type SpeciesName;

        HINLINE RandomNrForMonteCarlo(uint32_t currentStep) : isInitialized(false)
        {
            typedef typename SpeciesType::FrameType FrameType;

            mpi::SeedPerRank<simDim> seedPerRank;
            seed = seedPerRank(GlobalSeed()(), PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid());
            seed ^= IONIZATION_SEED;

            const uint32_t numSlides = MovingWindow::getInstance( ).getSlideCounter( currentStep );
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            localCells = subGrid.getLocalDomain().size;
            DataSpace<simDim> totalGpuOffset = subGrid.getLocalDomain( ).offset;
            totalGpuOffset.y( ) += numSlides * localCells.y( );
        }

        DINLINE void init(const DataSpace<simDim>& totalCellOffset)
        {
            if (!isInitialized)
            {
                const DataSpace<simDim> localCellIdx(totalCellOffset - totalGpuOffset);
                const uint32_t cellIdx = DataSpaceOperations<simDim>::map(
                                                                          localCells,
                                                                          localCellIdx);
                rng = nvrng::create(rngMethods::Xor(seed, cellIdx), rngDistributions::Uniform_float());
                isInitialized = true;
            }
        }

        DINLINE float_X operator()()
        {
            return rng();
        }

        private:
            typedef PMacc::nvidia::rng::RNG<rngMethods::Xor, rngDistributions::Uniform_float> RngType;

            PMACC_ALIGN(rng, RngType);
            PMACC_ALIGN(isInitialized, bool);
            PMACC_ALIGN(seed, uint32_t);
            PMACC_ALIGN(localCells, DataSpace<simDim>);
            PMACC_ALIGN(totalGpuOffset, DataSpace<simDim>);
    };

} // namespace ionization

} // namespace particles

} // namespace picongpu

