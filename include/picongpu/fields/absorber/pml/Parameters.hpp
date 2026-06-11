/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
