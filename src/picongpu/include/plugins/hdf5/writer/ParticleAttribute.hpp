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

using namespace splash;
namespace bmpl = boost::mpl;

/** write attribute of a particle to hdf5 file
 * 
 * @tparam T_Identifier identifier of a particle attribute
 */
template< typename T_Identifier>
struct ParticleAttribute
{

    /** write attribute to hdf5 file
     * 
     * @param params wrapped params with domainwriter, ...
     * @param frame frame with all particles 
     * @param prefix a name prefix for hdf5 attribute (is combined to: prefix_nameOfAttribute)
     * @param simOffset offset from window origin of thedomain
     * @param localSize local domain size 
     */
    template<typename FrameType>
    HINLINE void operator()(
                            const RefWrapper<ThreadParams*> params,
                            const RefWrapper<FrameType> frame,
                            const std::string prefix,
                            const DomainInformation domInfo,
                            const size_t elements)
    {

        typedef T_Identifier Identifier;
        typedef typename Identifier::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        typedef typename PICToSplash<ComponentType>::type SplashType;

        log<picLog::INPUT_OUTPUT > ("HDF5: write species attribute: %1%") % Identifier::getName();

        SplashType splashType;
        const std::string name_lookup[] = {"x", "y", "z"};

        std::vector<double> unit = Unit<T_Identifier>::get();

        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset = DataSpace<simDim>(
                                                                0,
                                                                params.get()->window.slides * params.get()->window.localFullSize.y(),
                                                                0);

        Dimensions splashDomainOffset(0, 0, 0);
        Dimensions splashGlobalDomainOffset(0, 0, 0);

        Dimensions splashDomainSize(1, 1, 1);
        Dimensions splashGlobalDomainSize(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashDomainOffset[d] = domInfo.domainOffset[d] + globalSlideOffset[d];
            splashGlobalDomainOffset[d] = domInfo.globalDomainOffset[d] + globalSlideOffset[d];
            splashGlobalDomainSize[d] = domInfo.globalDomainSize[d];
            splashDomainSize[d] = domInfo.domainSize[d];
        }

        for (uint32_t d = 0; d < components; d++)
        {
            std::stringstream str;
            str << prefix << "_" << T_Identifier::getName();
            if (components > 1)
                str << "_" << name_lookup[d];

            ValueType* dataPtr = frame.get().getIdentifier(Identifier()).getPointer();
            /** \todo fix me (libSplash issue #42
             *  Splash not accept NULL pointer, but we know that we have zero data to write
             * but we need to create the attribute, thus we set dataPtr to 1.
             * The reason why we have a data pointer is, thaat we not create memory if we need no memory.
             */
            //   if(dataPtr==NULL)
            //      dataPtr+=1;
            /** \todo fix me
             * this method not support to write empty attrbutes
             * (after we have fixed this in splash we can use this call and not appendDomain
            params.get()->dataCollector->writeDomain(params.get()->currentStep, 
                                                     splashType, 
                                                     1u, 
                                                     Dimensions(elements*components, 1, 1),
                                                     Dimensions(components, 1, 1),
                                                     Dimensions(elements, 1, 1),
                                                     Dimensions(d, 0, 0),
                                                     str.str().c_str(), 
                                                     domain_offset, 
                                                     domain_size, 
                                                     Dimensions(0, 0, 0), 
                                                     sim_global_size, 
                                                     DomainCollector::PolyType,
                                                     dataPtr);
             */
            params.get()->dataCollector->appendDomain(params.get()->currentStep,
                                                      splashType,
                                                      elements,
                                                      d,
                                                      components,
                                                      str.str().c_str(),
                                                      splashDomainOffset,
                                                      splashDomainSize,
                                                      splashGlobalDomainOffset,
                                                      splashGlobalDomainSize,
                                                      dataPtr);

            ColTypeDouble ctDouble;
            if (unit.size() >= (d + 1))
                params.get()->dataCollector->writeAttribute(params.get()->currentStep,
                                                            ctDouble, str.str().c_str(), "sim_unit", &(unit.at(d)));


        }
        log<picLog::INPUT_OUTPUT > ("HDF5: Finish write species attribute: %1%") % Identifier::getName();
    }

};

} //namspace hdf5

} //namespace picongpu

