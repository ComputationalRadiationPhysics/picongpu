/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2014-2024 Alexander Debus, Axel Huebl, Sergei Bastrakov
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
#include "picongpu/fields/background/templates/twtstight/TWTSTight.hpp"

#include <pmacc/math/Complex.hpp>
#include <pmacc/types.hpp>

namespace picongpu::templates::twtstight
{
    using EField = TWTSTight<FieldE>;

    /** Calculate the Ex(r,t) field here
     *
     * @param pos Spatial position of the target field.
     * @param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    template<>
    HDINLINE float_T EField::calcTWTSFieldX(float3_64 const& pos, float_64 const time) const
    {
        auto const minimalCoordinates = defineMinimalCoordinates(pos, time);

        /* To avoid underflows in computation, fields are set to zero
         * before and after the respective TWTS pulse envelope.
         */
        if(isOutsideTWTSEnvelope(minimalCoordinates))
            return float_T(0.0);

        auto const commonHelperVariables = defineCommonHelperVariables(minimalCoordinates);

        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        auto const [x, y, z, t] = minimalCoordinates;
        auto const [tanPhi, cotPhi, sinPhi_2, cosPhi_2, sinPolAngle, cosPolAngle, sin2Phi] = trigonometryShortcuts;
        auto const [floatHelpers, complexHelpers] = commonHelperVariables;
        auto const [x2, tauG2, psi0, w02, beta02, nu, xi, besselI0const] = floatHelpers;
        auto const [Xm, rhom, Xm2, rhom2, besselJ0const, besselJ1const] = complexHelpers;

        auto const zeroOrder = defineTWTSEnvelope(minimalCoordinates, commonHelperVariables);

        complex_T const result
            = (complex_T(0, 0.25) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
               * (k * rhom * besselJ0const
                      * ((rhom2 - x2 + x * Xm * cosPhi) * (sinPolAngle * sinPhi_2)
                         + cosPolAngle * (rhom2 + rhom2 * cosPhi_2 - x2 * sinPhi_2 - x * cosPhi * sinPhi_2 * Xm))
                  + besselJ1const * sinPhi
                        * (+sinPolAngle
                               * (-rhom2 + float_T(2.0) * x2 - I * rhom2 * Xm * (k * sinPhi)
                                  + x * cosPhi * (float_T(-2.0) * Xm - I * rhom2 * (k * sinPhi)))
                           + cosPolAngle
                                 * (-rhom2 + float_T(2.0) * x2 + I * rhom2 * Xm * (k * sinPhi)
                                    + x * cosPhi * (float_T(+2.0) * Xm + I * rhom2 * (k * sinPhi)))))
               * psi0)
              / (rhom * rhom2 * besselI0const);

        /* A 180deg-rotation of the field vector around the y-axis
         * leads to a sign flip in the x- and z- components, respectively.
         * This is implemented by multiplying the result by "phiPositive".
         */
        return phiPositive * result.real();
    }

    /** Calculate the Ey(r,t) field here
     *
     * @param pos Spatial position of the target field.
     * @param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    template<>
    HDINLINE float_T EField::calcTWTSFieldY(float3_64 const& pos, float_64 const time) const
    {
        auto const minimalCoordinates = defineMinimalCoordinates(pos, time);

        /* To avoid underflows in computation, fields are set to zero
         * before and after the respective TWTS pulse envelope.
         */
        if(isOutsideTWTSEnvelope(minimalCoordinates))
            return float_T(0.0);

        auto const commonHelperVariables = defineCommonHelperVariables(minimalCoordinates);

        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        auto const [x, y, z, t] = minimalCoordinates;
        auto const [tanPhi, cotPhi, sinPhi_2, cosPhi_2, sinPolAngle, cosPolAngle, sin2Phi] = trigonometryShortcuts;
        auto const [floatHelpers, complexHelpers] = commonHelperVariables;
        auto const [x2, tauG2, psi0, w02, beta02, nu, xi, besselI0const] = floatHelpers;
        auto const [Xm, rhom, Xm2, rhom2, besselJ0const, besselJ1const] = complexHelpers;

        auto const zeroOrder = defineTWTSEnvelope(minimalCoordinates, commonHelperVariables);

        complex_T const result = (math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder * (k * sinPhi)
                                  * (besselJ1const
                                         * (cosPolAngle * (Xm - float_T(2.0) * x * cosPhi - Xm * cosPhi_2)
                                            + (float_T(1.0) + cosPhi_2) * sinPolAngle * Xm)
                                     + I * rhom * besselJ0const * ((cosPolAngle - sinPolAngle) * sinPhi_2))
                                  * psi0)
                                 / (float_T(4.0) * besselI0const * rhom);

        return result.real();
    }

    /** Calculate the Ez(r,t) field here
     *
     * @param pos Spatial position of the target field.
     * @param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    template<>
    HDINLINE float_T EField::calcTWTSFieldZ(float3_64 const& pos, float_64 const time) const
    {
        auto const minimalCoordinates = defineMinimalCoordinates(pos, time);

        /* To avoid underflows in computation, fields are set to zero
         * before and after the respective TWTS pulse envelope.
         */
        if(isOutsideTWTSEnvelope(minimalCoordinates))
            return float_T(0.0);

        auto const commonHelperVariables = defineCommonHelperVariables(minimalCoordinates);

        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        auto const [x, y, z, t] = minimalCoordinates;
        auto const [tanPhi, cotPhi, sinPhi_2, cosPhi_2, sinPolAngle, cosPolAngle, sin2Phi] = trigonometryShortcuts;
        auto const [floatHelpers, complexHelpers] = commonHelperVariables;
        auto const [x2, tauG2, psi0, w02, beta02, nu, xi, besselI0const] = floatHelpers;
        auto const [Xm, rhom, Xm2, rhom2, besselJ0const, besselJ1const] = complexHelpers;

        auto const zeroOrder = defineTWTSEnvelope(minimalCoordinates, commonHelperVariables);

        complex_T const result
            = (complex_T(0, 0.125) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
               * (float_T(2.0) * k * rhom * besselJ0const
                      * (x * (cosPolAngle + sinPolAngle) * sinPhi_2 * Xm
                         + cosPhi
                               * (cosPolAngle * sinPhi_2 * Xm2
                                  + sinPolAngle * (float_T(2.0) * rhom2 - Xm2 * sinPhi_2)))
                  + besselJ1const * sinPhi
                        * (cosPolAngle
                               * (float_T(-4.0) * x * Xm + float_T(2.0) * cosPhi * (rhom2 - float_T(2.0) * Xm2)
                                  + complex_T(0, 2) * rhom2 * (x - Xm * cosPhi) * (k * sinPhi))
                           + sinPolAngle
                                 * (float_T(-4.0) * x * Xm - float_T(2.0) * cosPhi * (rhom2 - float_T(2.0) * Xm2)
                                    - complex_T(0, 2) * rhom2 * (k * x * sinPhi) + I * rhom2 * Xm * (k * sin2Phi))))
               * psi0)
              / (besselI0const * rhom * rhom2);

        /* A 180deg-rotation of the field vector around the y-axis
         * leads to a sign flip in the x- and z- components, respectively.
         * This is implemented by multiplying the result by "phiPositive".
         */
        return phiPositive * result.real();
    }

} /* namespace picongpu::templates::twtstight */
