/**
 * Copyright 2014-2015 Felix Schmitt, Axel Huebl
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
#include "traits/Resolve.hpp"

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
                            ThreadParams* params,
                            const std::string subGroup,
                            const uint64_t elements,
                            const uint64_t globalElements,
                            const uint64_t globalOffset)
    {

        typedef T_Identifier Identifier;
        typedef typename PMacc::traits::Resolve<Identifier>::type::type ValueType;
        const uint32_t components = GetNComponents<ValueType>::value;
        typedef typename GetComponentsType<ValueType>::type ComponentType;

        typedef typename traits::PICToAdios<ComponentType> AdiosType;

        params->adiosGroupSize += elements * components * sizeof(ComponentType);

        /* define adios var for particle attribute */
        AdiosType adiosType;
        const std::string name_lookup[] = {"x", "y", "z"};

        std::vector<float_64> unit = Unit<Identifier>::get();

        for (uint32_t d = 0; d < components; d++)
        {
            std::stringstream datasetName;
            datasetName << params->adiosBasePath << ADIOS_PATH_PARTICLES <<
                    subGroup << T_Identifier::getName();
            if (components > 1)
                datasetName << "/" << name_lookup[d];

            const char* path = NULL;
            int64_t adiosParticleAttrId = defineAdiosVar<DIM1>(
                params->adiosGroupHandle,
                datasetName.str().c_str(),
                path,
                adiosType.type,
                PMacc::math::UInt64<DIM1>(elements),
                PMacc::math::UInt64<DIM1>(globalElements),
                PMacc::math::UInt64<DIM1>(globalOffset),
                true,
                params->adiosCompression);

            params->adiosParticleAttrVarIds.push_back(adiosParticleAttrId);

            /* already add the sim_unit attribute so `adios_group_size` calculates
             * the reservation for the buffer correctly */
            AdiosDoubleType adiosDoubleType;

            /* check if this attribute actually has a unit (unit.size() == 0 is no unit) */
            if (unit.size() >= (d + 1))
                ADIOS_CMD(adios_define_attribute_byvalue(params->adiosGroupHandle,
                          "sim_unit", datasetName.str().c_str(),
                          adiosDoubleType.type, 1, (void*)&unit.at(d) ));
        }


    }

};

} //namspace adios

} //namespace picongpu

