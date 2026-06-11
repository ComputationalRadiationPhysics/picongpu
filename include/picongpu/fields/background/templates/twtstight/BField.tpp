/*
 * SPDX-FileCopyrightText: Alexander Debus, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/background/templates/twtstight/TWTSTight.hpp"

#include <pmacc/math/Complex.hpp>
#include <pmacc/types.hpp>

namespace picongpu::templates::twtstight
{
    using BField = TWTSTight<FieldB>;

    /** Calculate the Bx(r,t) field
     *
     * @param pos Spatial position of the target field.
     * @param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    template<>
    HDINLINE float_T BField::calcTWTSFieldX(float3_64 const& pos, float_64 const time) const
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
            = (complex_T(0, -0.25) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
               * (k * rhom * besselJ0const
                      * (cosPolAngle * (-rhom2 + x2 + x * cosPhi * Xm) * sinPhi_2
                         - sinPolAngle * (rhom2 + rhom2 * cosPhi_2 - x2 * sinPhi_2 + x * cosPhi * sinPhi_2 * Xm))
                  + besselJ1const
                        * (cosPolAngle * sinPhi
                               * (rhom2 - float_T(2.0) * x2 + I * Xm * rhom2 * (k * sinPhi)
                                  + x * cosPhi * (float_T(-2.0) * Xm - I * rhom2 * (k * sinPhi)))
                           + sinPolAngle
                                 * ((rhom2 - float_T(2.0) * x2) * sinPhi
                                    + I * rhom2 * (-Xm + x * cosPhi) * (k * sinPhi_2) + x * sin2Phi * Xm)))
               * psi0)
              / (cspeed * besselI0const * rhom * rhom2);

        /* A 180deg-rotation of the field vector around the y-axis
         * leads to a sign flip in the x- and z- components, respectively.
         * This is implemented by multiplying the result by "phiPositive".
         */
        return phiPositive * result.real() / sim.unit.speed();
    }

    /** Calculate the By(r,t) field here
     *
     * @param pos Spatial position of the target field.
     * @param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    template<>
    HDINLINE float_T BField::calcTWTSFieldY(float3_64 const& pos, float_64 const time) const
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
                                  * (-(besselJ1const
                                       * (cosPolAngle * (float_T(1.0) + cosPhi_2) * Xm
                                          + (Xm + float_T(2.0) * x * cosPhi - Xm * cosPhi_2) * sinPolAngle))
                                     + I * rhom * besselJ0const * (cosPolAngle - sinPolAngle) * sinPhi_2)
                                  * psi0)
                                 / (float_T(4.0) * cspeed * besselI0const * rhom);

        return result.real() / sim.unit.speed();
    }

    /** Calculate the Bz(r,t) field
     *
     * @param pos Spatial position of the target field.
     * @param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    template<>
    HDINLINE float_T BField::calcTWTSFieldZ(float3_64 const& pos, float_64 const time) const
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
            = (complex_T(0, -0.25) * math::exp(I * (omega0 * t - k * y * cosPhi)) * zeroOrder
               * (besselJ1const * sinPhi
                      * (sinPolAngle
                             * (x * (float_T(2.0) * Xm - I * (k * rhom2 * sinPhi))
                                + cosPhi * (rhom2 - float_T(2.0) * Xm2 - I * Xm * (k * rhom2 * sinPhi)))
                         + cosPolAngle
                               * (x * (float_T(2.0) * Xm + I * (k * rhom2 * sinPhi))
                                  + cosPhi * (-rhom2 + float_T(2.0) * Xm2 + I * Xm * (k * rhom2 * sinPhi))))
                  + k * rhom * besselJ0const
                        * (Xm * (-x + Xm * cosPhi) * (sinPolAngle * sinPhi_2)
                           - cosPolAngle * (x * sinPhi_2 * Xm + cosPhi * (float_T(-2.0) * rhom2 + Xm2 * sinPhi_2))))
               * psi0)
              / (cspeed * besselI0const * rhom * rhom2);

        /* A 180deg-rotation of the field vector around the y-axis
         * leads to a sign flip in the x- and z- components, respectively.
         * This is implemented by multiplying the result by "phiPositive".
         */
        return phiPositive * result.real() / sim.unit.speed();
    }

} /* namespace picongpu::templates::twtstight */
