/**
 * Copyright 2015 Heiko Burau, Marco Garten
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

#include "compileTime/conversion/TypeToPointerPair.hpp"
#include "memory/boxes/DataBox.hpp"

namespace picongpu
{
namespace particles
{
namespace creation
{

/** This functor is a base class for an actual particle creator.
 *
 * \tparam T_DestSpecies electron species to be created
 */
template<typename T_SourceSpecies, typename T_TargetSpecies>
struct CreatorBase
{

    typedef T_SourceSpecies SourceSpecies;
    typedef T_TargetSpecies TargetSpecies;

    typedef typename SourceSpecies::FrameType FrameType;

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

    //BlockArea BlockDescription;

protected:

    typedef MappingDesc::SuperCellSize TVec;

    typedef FieldE::ValueType ValueType_E;
    typedef FieldB::ValueType ValueType_B;
    /* global memory EM-field device databoxes */
    PMACC_ALIGN(eBox, FieldE::DataBoxType);
    PMACC_ALIGN(bBox, FieldB::DataBoxType);
    /* shared memory EM-field device databoxes */
    PMACC_ALIGN(cachedE, DataBox<SharedBox<ValueType_E, typename BlockArea::FullSuperCellSize,1> >);
    PMACC_ALIGN(cachedB, DataBox<SharedBox<ValueType_B, typename BlockArea::FullSuperCellSize,0> >);

    /* host constructor initializing member : random number generator */
    CreatorBase()
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        /* initialize pointers on host-side E-(B-)field databoxes */
        FieldE* fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));
        FieldB* fieldB = &(dc.getData<FieldB > (FieldB::getName(), true));
        /* initialize device-side E-(B-)field databoxes */
        eBox = fieldE->getDeviceDataBox();
        bBox = fieldB->getDeviceDataBox();
    }

    /** Initialization function on device
     *
     * \brief Cache EM-fields on device
     *         and initialize possible prerequisites for ionization, like e.g. random number generator.
     *
     * This function will be called inline on the device which must happen BEFORE threads diverge
     * during loop execution. The reason for this is the `__syncthreads()` call which is necessary after
     * initializing the E-/B-field shared boxes in shared memory.
     */
    DINLINE void init(const DataSpace<simDim>& blockCell, const int& linearThreadIdx, const DataSpace<simDim>& localCellOffset)
    {

        /* caching of E and B fields */
        cachedB = CachedBox::create < 0, ValueType_B > (BlockArea());
        cachedE = CachedBox::create < 1, ValueType_E > (BlockArea());

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

        /* wait for shared memory to be initialized */
        __syncthreads();
    }

    /** Does EM-field-interpolation at the particle's position
     * @param particle particle object
     * @param eField result of the bField interpolation
     * @param bField result of the eField interpolation
     */
    template<typename Particle>
    DINLINE void getFieldsForParticle(const Particle& particle, ValueType_E& eField, ValueType_B& bField) const
    {
        /* type of PIC-scheme cell */
        typedef typename fieldSolver::NumericalCellType NumericalCellType;

        /* particle position, used for field-to-particle interpolation */
        floatD_X pos = particle[position_];
        const int particleCellIdx = particle[localCellIdx_];
        /* multi-dim coordinate of the local cell inside the super cell */
        DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec > (particleCellIdx));
        /* interpolation of E- */
        eField = Field2ParticleInterpolation()
            (cachedE.shift(localCell).toCursor(), pos, NumericalCellType::getEFieldPosition());
        /*                     and B-field on the particle position */
        bField = Field2ParticleInterpolation()
            (cachedB.shift(localCell).toCursor(), pos, NumericalCellType::getBFieldPosition());
    }

    /** Return the number of target particles to be created from each source particle.
     *
     * Called for each frame of the source species. This is an abstract function indented
     * to be implemented in a derived class.
     *
     * @param sourceFrame Frame of the source species
     * @param localIdx Index of the source particle within frame
     * @return number of particle to be created from each source particle
     */
    DINLINE unsigned int numNewParticles(FrameType& sourceFrame, int localIdx)
    {
        return 0;
    }

    /** Functor implementation.
     *
     * Called once for each single particle creation. This is an abstract function indented
     * to be implemented in a derived class.
     *
     * \tparam SourceParticle type of the particle which induces particle creation
     * \tparam TargetParticle type of the particle that will be created
     */
    template<typename SourceParticle, typename TargetParticle>
    DINLINE void operator()(SourceParticle& sourceParticle, TargetParticle& targetParticle)
    {

    }
};

} // namespace creation
} // namespace particles
} // namespace picongpu
