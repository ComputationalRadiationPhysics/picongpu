/*
 * SPDX-FileCopyrightText: Axel Huebl, Rene Widera, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/absorber/AbsorberImpl.hpp"
#include "picongpu/fields/absorber/exponential/Exponential.kernel"
#include "picongpu/fields/absorber/param.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
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
