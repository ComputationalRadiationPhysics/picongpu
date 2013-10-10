/**
 * Copyright 2013 René Widera
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
#include "traits/PICToSplash.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"


namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

using namespace DCollector;
namespace bmpl = boost::mpl;

template< typename T_Identifier>
struct ParticleAttribute
{

    template<typename Space, typename FrameType>
    HINLINE void operator()(
                            const RefWrapper<ThreadParams*> params,
                            const RefWrapper<FrameType> frame,
                            const std::string prefix,
                            const Space simOffset,
                            const Space simSize,
                            const size_t elements)
    {

        typedef T_Identifier Identifier;
        typedef typename Identifier::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        typedef typename PICToSplash<ComponentType>::type SplashType;

        log<picLog::INPUT_OUTPUT > ("HDF5: write species attribute: %1%") % Identifier::getName();

        SplashType spashType;
        const std::string name_lookup[] = {"x", "y", "z"};

        std::vector<double> unit = Unit<T_Identifier>::get();

        DataSpace<simDim> field_no_guard = simSize;
        DataSpace<simDim> global_sim_size = params.get()->window.globalSimulationSize;

        DCollector::Dimensions domain_offset(0, 0, 0);
        DCollector::Dimensions domain_size(1, 1, 1);

        ///\todo this might be deprecated
        DCollector::Dimensions sim_global_offset(0, 0, 0);
        DCollector::Dimensions sim_global_size(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            sim_global_size[d] = global_sim_size[d];
            if (simOffset[d] > 0)
            {
                sim_global_offset[d] = simOffset[d];
                domain_offset[d] = simOffset[d];
            }
            domain_size[d] = field_no_guard[d];
        }

        for (uint32_t d = 0; d < components; d++)
        {
            std::stringstream str;
            str << prefix << "_" << T_Identifier::getName();
            if (components > 1)
                str << "_" << name_lookup[d];

            ValueType* dataPtr = frame.get().getIdentifier(Identifier()).getPointer();
            /** \todo fix me
             *  Splash not accept NULL pointer, but we know that we have zero data to write
             * but we need to create the attribute, thus we set dataPtr to 1.
             * The reason why we have a data pointer is, thaat we not create memory if we need no memory.
             */
            if(dataPtr==NULL)
                dataPtr+=1;
            /** \todo fix me
             * this method not support to write empty attrbutes
             * (after we have fixed this in splash we can use this call and not appendDomain
            params.get()->dataCollector->writeDomain(params.get()->currentStep, 
                                                     spashType, 
                                                     1u, 
                                                     DCollector::Dimensions(elements*components, 1, 1),
                                                     DCollector::Dimensions(components, 1, 1),
                                                     DCollector::Dimensions(elements, 1, 1),
                                                     DCollector::Dimensions(d, 0, 0),
                                                     str.str().c_str(), 
                                                     domain_offset, 
                                                     domain_size, 
                                                     Dimensions(0, 0, 0), 
                                                     sim_global_size, 
                                                     DomainCollector::PolyType,
                                                     dataPtr);
             */
            params.get()->dataCollector->appendDomain(params.get()->currentStep,
                                                spashType,
                                                elements,
                                                d,
                                                components,
                                                str.str().c_str(),
                                                domain_offset,
                                                domain_size,
                                                Dimensions(0, 0, 0), /** \todo set to moving window offset */
                                                sim_global_size,
                                                dataPtr);


            DCollector::ColTypeDouble ctDouble;
            if (unit.size() >= (d + 1))
                params.get()->dataCollector->writeAttribute(params.get()->currentStep,
                                                            ctDouble, str.str().c_str(), "sim_unit", &(unit.at(d)));
            params.get()->dataCollector->writeAttribute(params.get()->currentStep,
                                                        DCollector::ColTypeDim(), str.str().c_str(),
                                                        "sim_global_offset", sim_global_offset.getPointer());


        }
        log<picLog::INPUT_OUTPUT > ("HDF5: Finish write species attribute: %1%") % Identifier::getName();
    }

};

} //namspace hdf5

} //namepsace picongpu

