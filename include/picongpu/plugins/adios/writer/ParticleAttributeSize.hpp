/* Copyright 2014-2021 Felix Schmitt, Axel Huebl
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/adios/ADIOSWriter.def"
#include "picongpu/plugins/misc/ComponentNames.hpp"
#include "picongpu/traits/PICToAdios.hpp"
#include "picongpu/traits/PICToOpenPMD.hpp"
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/assert.hpp>

namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;


        /** collect size of a particle attribute
         *
         * @tparam T_Identifier identifier of a particle attribute
         */
        template<typename T_Identifier>
        struct ParticleAttributeSize
        {
            /** collect size of attribute
             *
             * @param params wrapped params
             * @param elements number of particles for this attribute
             */
            HINLINE void operator()(
                ThreadParams* params,
                const std::string speciesGroup,
                const uint64_t elements,
                const uint64_t globalElements,
                const uint64_t globalOffset)
            {
                typedef T_Identifier Identifier;
                typedef typename pmacc::traits::Resolve<Identifier>::type::type ValueType;
                const uint32_t components = GetNComponents<ValueType>::value;
                typedef typename GetComponentsType<ValueType>::type ComponentType;

                params->adiosGroupSize += elements * components * sizeof(ComponentType);

                /* define adios var for particle attribute */
                PICToAdios<ComponentType> adiosType;
                PICToAdios<float_X> adiosFloatXType;
                PICToAdios<float_64> adiosDoubleType;
                PICToAdios<uint32_t> adiosUInt32Type;

                const auto componentNames = plugins::misc::getComponentNames(components);

                OpenPMDName<T_Identifier> openPMDName;
                const std::string recordPath(
                    params->adiosBasePath + std::string(ADIOS_PATH_PARTICLES) + speciesGroup + openPMDName());

                // get the SI scaling, dimensionality and weighting of the attribute
                OpenPMDUnit<T_Identifier> openPMDUnit;
                std::vector<float_64> unit = openPMDUnit();
                OpenPMDUnitDimension<T_Identifier> openPMDUnitDimension;
                std::vector<float_64> unitDimension = openPMDUnitDimension();
                const bool macroWeightedBool = MacroWeighted<T_Identifier>::get();
                const uint32_t macroWeighted = (macroWeightedBool ? 1 : 0);
                const float_64 weightingPower = WeightingPower<T_Identifier>::get();

                PMACC_ASSERT(unit.size() == components); // unitSI for each component
                PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units

                for(uint32_t d = 0; d < components; d++)
                {
                    std::stringstream datasetName;
                    datasetName << recordPath;
                    if(components > 1)
                        datasetName << "/" << componentNames[d];

                    const char* path = nullptr;
                    int64_t adiosParticleAttrId = defineAdiosVar<DIM1>(
                        params->adiosGroupHandle,
                        datasetName.str().c_str(),
                        path,
                        adiosType.type,
                        pmacc::math::UInt64<DIM1>(elements),
                        pmacc::math::UInt64<DIM1>(globalElements),
                        pmacc::math::UInt64<DIM1>(globalOffset),
                        true,
                        params->adiosCompression);

                    params->adiosParticleAttrVarIds.push_back(adiosParticleAttrId);

                    /* already add the unitSI and further attribute so `adios_group_size`
                     * calculates the reservation for the buffer correctly */

                    /* check if this attribute actually has a unit (unit.size() == 0 is no unit) */
                    if(unit.size() >= (d + 1))
                        ADIOS_CMD(adios_define_attribute_byvalue(
                            params->adiosGroupHandle,
                            "unitSI",
                            datasetName.str().c_str(),
                            adiosDoubleType.type,
                            1,
                            &unit.at(d)));
                }

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "unitDimension",
                    recordPath.c_str(),
                    adiosDoubleType.type,
                    7,
                    &(*unitDimension.begin())));

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "macroWeighted",
                    recordPath.c_str(),
                    adiosUInt32Type.type,
                    1,
                    (void*) &macroWeighted));

                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "weightingPower",
                    recordPath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &weightingPower));

                /** \todo check if always correct at this point, depends on attribute
                 *        and MW-solver/pusher implementation */
                const float_X timeOffset = 0.0;
                ADIOS_CMD(adios_define_attribute_byvalue(
                    params->adiosGroupHandle,
                    "timeOffset",
                    recordPath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &timeOffset));
            }
        };

    } // namespace adios

} // namespace picongpu
