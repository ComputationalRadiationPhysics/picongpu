/* Copyright 2014-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;

        /** write attribute of a particle to adios file
         *
         * @tparam T_Identifier identifier of a particle attribute
         */
        template<typename T_Identifier>
        struct ParticleAttribute
        {
            /** write attribute to adios file
             *
             * @param params wrapped params
             * @param elements elements of this attribute
             */
            template<typename FrameType>
            HINLINE void operator()(ThreadParams* params, FrameType& frame, const size_t elements)
            {
                typedef T_Identifier Identifier;
                typedef typename pmacc::traits::Resolve<Identifier>::type::type ValueType;
                const uint32_t components = GetNComponents<ValueType>::value;
                typedef typename GetComponentsType<ValueType>::type ComponentType;

                log<picLog::INPUT_OUTPUT>("ADIOS:  (begin) write species attribute: %1%") % Identifier::getName();

                ComponentType* tmpBfr = nullptr;

                if(elements > 0)
                    tmpBfr = new ComponentType[elements];

                for(uint32_t d = 0; d < components; d++)
                {
                    ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();

/* copy strided data from source to temporary buffer */
#pragma omp parallel for
                    for(size_t i = 0; i < elements; ++i)
                    {
                        tmpBfr[i] = ((ComponentType*) dataPtr)[d + i * components];
                    }

                    int64_t adiosAttributeVarId = *(params->adiosParticleAttrVarIds.begin());
                    params->adiosParticleAttrVarIds.pop_front();

                    ADIOS_CMD(adios_write_byid(params->adiosFileHandle, adiosAttributeVarId, tmpBfr));
                }

                __deleteArray(tmpBfr);

                log<picLog::INPUT_OUTPUT>("ADIOS:  ( end ) write species attribute: %1%") % Identifier::getName();
            }
        };

    } // namespace adios

} // namespace picongpu
