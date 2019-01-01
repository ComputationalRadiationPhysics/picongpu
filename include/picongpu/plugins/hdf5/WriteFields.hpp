/* Copyright 2014-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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

#include <pmacc/static_assert.hpp>
#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/hdf5/HDF5Writer.def"
#include "picongpu/plugins/hdf5/writer/Field.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>

#include <vector>


namespace picongpu
{

namespace hdf5
{

using namespace pmacc;
using namespace splash;

/**
 * Helper class to create a unit vector of type float_64
 */
class CreateUnit
{
public:
    template<typename UnitType>
    static std::vector<float_64> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<float_64> tmp(numComponents);
        for (uint32_t i = 0; i < numComponents; ++i)
            tmp[i] = unit[i];
        return tmp;
    }
};


/**
 * Write calculated fields to HDF5 file.
 *
 * @tparam T field class
 */
template< typename T >
class WriteFields
{
private:
    typedef typename T::ValueType ValueType;

    static std::vector<float_64> getUnit()
    {
        typedef typename T::UnitValueType UnitType;
        UnitType unit = T::getUnit();
        return CreateUnit::createUnit(unit, T::numComponents);
    }

public:

    HDINLINE void operator()(ThreadParams* params)
    {
#ifndef __CUDA_ARCH__
        DataConnector &dc = Environment<>::get().DataConnector();

        auto field = dc.get< T >( T::getName() );
        params->gridLayout = field->getGridLayout();

        // convert in a std::vector of std::vector format for writeField API
        const traits::FieldPosition<typename fields::Solver::NummericalCellType, T> fieldPos;

        std::vector<std::vector<float_X> > inCellPosition;
        for( uint32_t n = 0; n < T::numComponents; ++n )
        {
            std::vector<float_X> inCellPositonComponent;
            for( uint32_t d = 0; d < simDim; ++d )
                inCellPositonComponent.push_back( fieldPos()[n][d] );
            inCellPosition.push_back( inCellPositonComponent );
        }

        /** \todo check if always correct at this point, depends on solver
         *        implementation */
        const float_X timeOffset = 0.0;

        Field::writeField(params,
                          T::getName(),
                          getUnit(),
                          T::getUnitDimension(),
                          inCellPosition,
                          timeOffset,
                          field->getHostDataBox(),
                          ValueType());

        dc.releaseData( T::getName() );
#endif
    }

};

/** Calculate FieldTmp with given solver and particle species
 * and write them to hdf5.
 *
 * FieldTmp is calculated on device and than dumped to HDF5.
 *
 * @tparam Solver solver class for species
 * @tparam Species species/particles class
 */
template< typename Solver, typename Species >
class WriteFields<FieldTmpOperation<Solver, Species> >
{
public:
    /*
     * This is only a wrapper function to allow disable nvcc warnings.
     * Warning: calling a __host__ function from __host__ __device__
     * function.
     * Use of PMACC_NO_NVCC_HDWARNING is not possible if we call a virtual
     * method inside of the method were we disable the warnings.
     * Therefore we create this method and call a new method were we can
     * call virtual functions.
     */
    PMACC_NO_NVCC_HDWARNING
    HDINLINE void operator()(ThreadParams* tparam)
    {
        this->operator_impl(tparam);
    }

private:
    typedef typename FieldTmp::ValueType ValueType;

    /** Create a name for the hdf5 identifier.
     */
    static std::string getName()
    {
        return FieldTmpOperation<Solver, Species>::getName();
    }

    /** Get the unit for the result from the solver*/
    static std::vector<float_64> getUnit()
    {
        typedef typename FieldTmp::UnitValueType UnitType;
        UnitType unit = FieldTmp::getUnit<Solver>();
        const uint32_t components = GetNComponents<ValueType>::value;
        return CreateUnit::createUnit(unit, components);
    }

    HINLINE void operator_impl(ThreadParams* params)
    {
        DataConnector &dc = Environment<>::get().DataConnector();

        /*## update field ##*/

        /*load FieldTmp without copy data to host*/
        PMACC_CASSERT_MSG(
            _please_allocate_at_least_one_FieldTmp_in_memory_param,
            fieldTmpNumSlots > 0
        );
        auto fieldTmp = dc.get< FieldTmp >( FieldTmp::getUniqueId( 0 ), true );
        /*load particle without copy particle data to host*/
        auto speciesTmp = dc.get< Species >( Species::FrameType::getName(), true );

        fieldTmp->getGridBuffer().getDeviceBuffer().setValue(ValueType::create(0.0));
        /*run algorithm*/
        fieldTmp->template computeValue< CORE + BORDER, Solver >(*speciesTmp, params->currentStep);

        EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(fieldTmpEvent);
        /* copy data to host that we can write same to disk*/
        fieldTmp->getGridBuffer().deviceToHost();
        dc.releaseData( Species::FrameType::getName() );
        /*## finish update field ##*/

        /*wrap in a one-component vector for writeField API*/
        const traits::FieldPosition<typename fields::Solver::NummericalCellType, FieldTmp>
            fieldPos;

        std::vector<std::vector<float_X> > inCellPosition;
        std::vector<float_X> inCellPositonComponent;
        for( uint32_t d = 0; d < simDim; ++d )
            inCellPositonComponent.push_back( fieldPos()[0][d] );
        inCellPosition.push_back( inCellPositonComponent );

        /** \todo check if always correct at this point, depends on solver
         *        implementation */
        const float_X timeOffset = 0.0;

        params->gridLayout = fieldTmp->getGridLayout();
        /*write data to HDF5 file*/
        Field::writeField(params,
                          getName(),
                          getUnit(),
                          FieldTmp::getUnitDimension<Solver>(),
                          inCellPosition,
                          timeOffset,
                          fieldTmp->getHostDataBox(),
                          ValueType());

        dc.releaseData( FieldTmp::getUniqueId( 0 ) );

    }

};

} //namspace hdf5

} //namespace picongpu
