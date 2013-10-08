/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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
#include "simulation_types.hpp"
#include "plugins/hdf5/HDF5Writer.def"


namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace DCollector;
namespace bmpl = boost::mpl;

template< typename T_Identifier>
struct WriteAttribute
{

    template<typename Space>
    HDINLINE void operator()(std::string prefix, Space sim_offset, Space simSize)
    {
        const int dims = Space::dim;

        const std::string name_lookup[] = {"x", "y", "z"};

        /*DCollector::ColTypeDouble ctDouble;
        DataSpace<simDim> field_no_guard = sim_size;

        DCollector::Dimensions domain_offset(0, 0, 0);
        DCollector::Dimensions domain_size(1, 1, 1);

        ///\todo this might be deprecated
        DCollector::Dimensions sim_global_offset(0, 0, 0);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            if (sim_offset[d] > 0)
            {
                sim_global_offset[d] = sim_offset[d];
                domain_offset[d] = sim_offset[d];
            }
            domain_size[d] = field_no_guard[d];
        }

        for (uint32_t d = 0; d < dims; d++)
        {
            std::stringstream str;
            str << prefix << T_Identifier::getName();
            if (dims > 1)
                str << "_" << name_lookup[d];

            params->dataCollector->appendDomain(params->currentStep,
                                                colType,
                                                elements,
                                                d,
                                                dims,
                                                str.str().c_str(),
                                                domain_offset,
                                                domain_size,
                                                ptr);

            if (unit != NULL)
                params->dataCollector->writeAttribute(params->currentStep, ctDouble, str.str().c_str(), "sim_unit", &(unit[d]));
            params->dataCollector->writeAttribute(params->currentStep, DCollector::ColTypeDim(), str.str().c_str(), "sim_global_offset", &sim_global_offset);
        }
         */
    }

};

} //namspace hdf5

} //namepsace picongpu

