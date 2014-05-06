/**
 * Copyright 2014 Felix Schmitt
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
#include "plugins/adios/ADIOSWriter.def"
#include "traits/PICToAdios.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"

namespace picongpu
{

namespace adios
{
using namespace PMacc;



/** collect size of a particle attribute
 * 
 * @tparam T_Identifier identifier of a particle attribute
 */
template< typename T_Identifier>
struct ParticleAttributeSize
{
    /** collect size of attribute
     * 
     * @param params wrapped params
     * @param elements number of particles for this attribute 
     */
    HINLINE void operator()(
                            const RefWrapper<ThreadParams*> params,
                            const std::string subGroup,
                            const size_t elements,
                            const size_t globalElements,
                            const size_t globalOffset)
    {

        typedef T_Identifier Identifier;
        typedef typename Identifier::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        
        typedef typename traits::PICToAdios<ComponentType> AdiosType;

        params.get()->adiosGroupSize += elements * components * sizeof(ComponentType);
        
        /* define adios var for particle attribute */
        AdiosType adiosType;
        const std::string name_lookup[] = {"x", "y", "z"};
        
        for (uint32_t d = 0; d < components; d++)
        {
            std::stringstream datasetName;
            datasetName << params.get()->adiosBasePath << ADIOS_PATH_PARTICLES <<
                    subGroup << T_Identifier::getName();
            if (components > 1)
                datasetName << "/" << name_lookup[d];
            
            /* get string representations for sizes and offset */
            std::stringstream localSizeStr;
            localSizeStr << elements;

            std::stringstream globalSizeStr;
            globalSizeStr << globalElements;

            std::stringstream offsetStr;
            offsetStr << globalOffset;

            int64_t adiosParticleAttrId = adios_define_var(
                params.get()->adiosGroupHandle,
                datasetName.str().c_str(),
                NULL,
                adiosType.type,
                localSizeStr.str().c_str(),
                globalSizeStr.str().c_str(),
                offsetStr.str().c_str());

            params.get()->adiosParticleAttrVarIds.push_back(adiosParticleAttrId);
        }

        
    }

};

} //namspace adios

} //namespace picongpu

