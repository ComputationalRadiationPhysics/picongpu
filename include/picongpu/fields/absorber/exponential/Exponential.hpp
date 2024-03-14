/* Copyright 2013-2023 Axel Huebl, Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/absorber/exponential/Exponential.kernel"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

#include <cstdint>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            namespace exponential
            {
                /** Exponential damping field absorber implementation
                 *
                 * Implements absorption.
                 */
                class ExponentialImpl : public AbsorberImpl
                {
                public:
                    /** Create exponential absorber implementation instance
                     *
                     * @param cellDescription mapping for kernels
                     */
                    ExponentialImpl(MappingDesc const cellDescription)
                        : AbsorberImpl(Absorber::Kind::Exponential, cellDescription)
                    {
                    }

                    /** Apply absorber to the given field
                     *
                     * @tparam BoxedMemory field box type
                     *
                     * @param currentStep current time iteration
                     * @param deviceBox field box
                     */
                    template<class BoxedMemory>
                    void run(float_X currentStep, BoxedMemory deviceBox)
                    {
                        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
                        {
                            /* only call for planes: left right top bottom back front*/
                            if(FRONT % i == 0
                               && !(Environment<simDim>::get().GridController().getCommunicationMask().isSet(i)))
                            {
                                uint32_t direction = 0; /*set direction to X (default)*/
                                if(i >= BOTTOM && i <= TOP)
                                    direction = 1; /*set direction to Y*/
                                if(i >= BACK)
                                    direction = 2; /*set direction to Z*/

                                /* exchange mod 2 to find positive or negative direction
                                 * positive direction = 1
                                 * negative direction = 0
                                 */
                                uint32_t pos_or_neg = i % 2;

                                uint32_t thickness = numCells[direction][pos_or_neg];
                                float_X absorber_strength = STRENGTH[direction][pos_or_neg];

                                if(thickness == 0)
                                    continue; /*if the absorber has no thickness we check the next side*/


                                /* if sliding window is active we disable absorber on bottom side*/
                                if(MovingWindow::getInstance().isSlidingWindowActive(
                                       static_cast<uint32_t>(currentStep))
                                   && i == BOTTOM)
                                    continue;

                                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);

                                PMACC_LOCKSTEP_KERNEL(KernelAbsorbBorder{})
                                    .config(
                                        mapper.getGridDim(),
                                        SuperCellSize{})(deviceBox, thickness, absorber_strength, mapper);
                            }
                        }
                    }
                };

            } // namespace exponential
        } // namespace absorber
    } // namespace fields
} // namespace picongpu
