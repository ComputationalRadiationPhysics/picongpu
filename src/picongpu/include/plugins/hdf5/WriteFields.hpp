/**
 * Copyright 2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "types.h"
#include "simulation_types.hpp"
#include "plugins/hdf5/HDF5Writer.def"
#include "plugins/hdf5/writer/Field.hpp"

namespace picongpu
{

namespace hdf5
{

using namespace PMacc;
using namespace splash;

/**
 * Helper class to create a unit vector of type double
 */
class CreateUnit
{
public:
    template<typename UnitType>
    static std::vector<double> createUnit(UnitType unit, uint32_t numComponents)
    {
        std::vector<double> tmp(numComponents);
        for (uint i = 0; i < numComponents; ++i)
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

    static std::vector<double> getUnit()
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

        T* field = &(dc.getData<T > (T::getName()));
        params->gridLayout = field->getGridLayout();

        Field::writeField(params,
                          T::getName(),
                          getUnit(),
                          field->getHostDataBox(),
                          ValueType());

        dc.releaseData(T::getName());
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
        std::stringstream str;
        str << Solver().getName();
        str << "_";
        str << Species::FrameType::getName();
        return str.str();
    }

    /** Get the unit for the result from the solver*/
    static std::vector<double> getUnit()
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
        FieldTmp* fieldTmp = &(dc.getData<FieldTmp > (FieldTmp::getName(), true));
        /*load particle without copy particle data to host*/
        Species* speciesTmp = &(dc.getData<Species >(Species::FrameType::getName(), true));

        fieldTmp->getGridBuffer().getDeviceBuffer().setValue(ValueType(0.0));
        /*run algorithm*/
        fieldTmp->computeValue < CORE + BORDER, Solver > (*speciesTmp, params->currentStep);

        EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
        __setTransactionEvent(fieldTmpEvent);
        /* copy data to host that we can write same to disk*/
        fieldTmp->getGridBuffer().deviceToHost();
        dc.releaseData(Species::FrameType::getName());
        /*## finish update field ##*/


        params->gridLayout = fieldTmp->getGridLayout();
        /*write data to HDF5 file*/
        Field::writeField(params,
                          getName(),
                          getUnit(),
                          fieldTmp->getHostDataBox(),
                          ValueType());

        dc.releaseData(FieldTmp::getName());

    }

};

} //namspace hdf5

} //namespace picongpu
