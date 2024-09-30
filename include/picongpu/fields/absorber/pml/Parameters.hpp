/* Copyright 2019-2023 Sergei Bastrakov
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

#include "picongpu/defines.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            namespace pml
            {
                /** Parameters of PML, except thickness
                 *
                 * A detailed description and recommended ranges are given in fieldAbsorber.param,
                 * normalizations and unit conversions in fieldAbsorber.unitless.
                 */
                struct Parameters
                {
                    /** Default constructor setting all members to 0
                     *
                     * This constructor only exists for deferred initialization on the host side.
                     */
                    Parameters()
                        : normalizedSigmaMax(floatD_X::create(0.0_X))
                        , sigmaKappaGradingOrder(0.0_X)
                        , kappaMax(floatD_X::create(0.0_X))
                        , normalizedAlphaMax(floatD_X::create(0.0_X))
                        , alphaGradingOrder(0.0_X)
                    {
                    }

                    /** Max value of artificial electric conductivity
                     *
                     * Components correspond to directions. Normalized, so that
                     * normalizedSigma = sigma / eps0 = sigma* / mue0.
                     * Unit: 1/unit_time in PIC units
                     */
                    floatD_X normalizedSigmaMax;

                    /** Order of polynomial growth of sigma and kappa
                     *
                     * The growth is from PML internal boundary to the external boundary.
                     * Sigma grows from 0, kappa from 1, both to their max values.
                     */
                    float_X sigmaKappaGradingOrder;

                    /** Max value of coordinate stretching coefficient
                     *
                     * Unitless.
                     */
                    floatD_X kappaMax;

                    /** Max value of complex frequency shift
                     *
                     * Components correspond to directions. Normalized by eps0.
                     * Unit: 1/unit_time in PIC units
                     */
                    floatD_X normalizedAlphaMax;

                    /** Order of polynomial growth of alpha
                     *
                     * The growth is from PML external boundary to the internal boundary.
                     * Grows from 0 to the max value.
                     */
                    float_X alphaGradingOrder;
                };

            } // namespace pml
        } // namespace absorber
    } // namespace fields
} // namespace picongpu
