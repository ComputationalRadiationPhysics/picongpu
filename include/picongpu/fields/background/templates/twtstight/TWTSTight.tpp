/*
 * SPDX-FileCopyrightText: Alexander Debus, Axel Huebl, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/fields/background/templates/twtstight/BField.tpp"
#include "picongpu/fields/background/templates/twtstight/EField.tpp"
#include "picongpu/fields/background/templates/twtstight/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/twtstight/TWTSTight.hpp"
#include "picongpu/fields/background/templates/twtstight/getFieldPositions_SI.tpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/types.hpp>

namespace picongpu::templates::twtstight
{
    /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device (--> HINLINE).
             Since this is done on host (see fieldBackground.param), this is no problem.
    */
    template<typename T_Field>
    HINLINE TWTSTight<T_Field>::TWTSTight(
        float_64 const focus_y_SI,
        float_64 const wavelength_SI,
        float_64 const pulselength_SI,
        float_64 const w_x_SI,
        float_64 const phi,
        float_64 const beta_0,
        float_64 const tdelay_user_SI,
        bool const auto_tdelay,
        float_64 const focus_z_offset_SI,
        float_64 const polAngle)
        : halfSimSize(Environment<simDim>::get().SubGrid().getGlobalDomain().size / 2)
        , focus_y_SI(focus_y_SI)
        , wavelength_SI(wavelength_SI)
        , pulselength_SI(pulselength_SI)
        , w_x_SI(w_x_SI)
        , phi(phi)
        , phiPositive(phi < 0.0_X ? float_X(-1.0) : float_X(+1.0))
        , beta_0(beta_0)
        , tdelay_user_SI(tdelay_user_SI)
        , focus_z_offset_SI(focus_z_offset_SI)
        , dt(sim.si.getDt())
        , unit_length(sim.unit.length())
        , auto_tdelay(auto_tdelay)
        , tdelay(
              detail::getInitialTimeDelay_SI(
                  auto_tdelay,
                  tdelay_user_SI,
                  halfSimSize,
                  pulselength_SI,
                  focus_y_SI,
                  phi,
                  beta_0))
        , polAngle(polAngle)
        , I(complex_T(0, 1))
        , basicTWTSHelperVariables(defineBasicHelperVariables(phi, beta_0, wavelength_SI, pulselength_SI, w_x_SI))
        , trigonometryShortcuts(defineTrigonometryShortcuts(basicTWTSHelperVariables, polAngle))
    {
    }

    template<typename T_Field>
    HDINLINE float3_X TWTSTight<T_Field>::getTWTSField_Normalized(
        pmacc::math::Vector<floatD_64, detail::numComponents> const& fieldPositions_SI,
        float_64 const time) const
    {
        if constexpr(simDim == DIM3)
        {
            /* E- or B-field normalized to the peak amplitude. */
            return precisionCast<float_X>(float3_T(
                calcTWTSFieldX(fieldPositions_SI[0], time),
                calcTWTSFieldY(fieldPositions_SI[1], time),
                calcTWTSFieldZ(fieldPositions_SI[2], time)));
        }
        if constexpr(simDim == DIM2)
        {
            using PosVecVec = pmacc::math::Vector<float3_64, detail::numComponents>;
            PosVecVec pos(PosVecVec::create(float3_64::create(0.0)));
            for(uint32_t k = 0; k < detail::numComponents; ++k)
            {
                /* 2D (y,z) vectors are mapped on 3D (x,y,z) vectors. */
                for(uint32_t i = 0; i < simDim; ++i)
                {
                    pos[k][i + 1] = fieldPositions_SI[k][i];
                }
            }
            /* E- or B-field normalized to the peak amplitude. */
            return precisionCast<float_X>(
                float3_T(calcTWTSFieldX(pos[0], time), calcTWTSFieldY(pos[1], time), calcTWTSFieldZ(pos[2], time)));
        }
        // We should never be here.
        else
            return float3_X(NAN);
    }

    template<typename T_Field>
    HDINLINE float3_X
    TWTSTight<T_Field>::operator()(DataSpace<simDim> const& cellIdx, uint32_t const currentStep) const
    {
        traits::FieldPosition<fields::YeeCell, T_Field> const fieldPos;
        return getValue(precisionCast<float_X>(cellIdx), fieldPos(), static_cast<float_X>(currentStep));
    }

    template<typename T_Field>
    HDINLINE float3_X TWTSTight<T_Field>::operator()(floatD_X const& cellIdx, float_X const currentStep) const
    {
        pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
        for(uint32_t component = 0; component < detail::numComponents; ++component)
            zeroShifts[component] = floatD_X::create(0.0);
        return getValue(cellIdx, zeroShifts, currentStep);
    }

    template<typename T_Field>
    HDINLINE float3_X TWTSTight<T_Field>::getValue(
        floatD_X const& cellIdx,
        pmacc::math::Vector<floatD_X, detail::numComponents> const& extraShifts,
        float_X const currentStep) const
    {
        float_64 const time_SI = float_64(currentStep) * dt - tdelay;

        pmacc::math::Vector<floatD_64, detail::numComponents> const fieldPositions_SI = detail::getFieldPositions_SI(
            cellIdx,
            halfSimSize,
            extraShifts,
            unit_length,
            focus_y_SI,
            focus_z_offset_SI,
            phi);

        /* Single TWTS-Pulse */
        return getTWTSField_Normalized(fieldPositions_SI, time_SI);
    }

    template<typename T_Field>
    template<uint32_t T_component>
    HDINLINE float_X TWTSTight<T_Field>::getComponent(floatD_X const& cellIdx, float_X const currentStep) const
    {
        // The optimized way is only implemented for 3D, fall back to full field calculation in 2d
        if constexpr(simDim == DIM3)
        {
            float_64 const time_SI = float_64(currentStep) * dt - tdelay;
            pmacc::math::Vector<floatD_X, detail::numComponents> zeroShifts;
            for(uint32_t component = 0; component < detail::numComponents; ++component)
                zeroShifts[component] = floatD_X::create(0.0);
            pmacc::math::Vector<floatD_64, detail::numComponents> const fieldPositions_SI
                = detail::getFieldPositions_SI(
                    cellIdx,
                    halfSimSize,
                    zeroShifts,
                    unit_length,
                    focus_y_SI,
                    focus_z_offset_SI,
                    phi);
            // Explicitly use a 3D vector so that this function compiles for 2D as well
            auto const pos = float3_64{
                fieldPositions_SI[T_component][0],
                fieldPositions_SI[T_component][1],
                fieldPositions_SI[T_component][2]};
            if constexpr(T_component == 0)
                return static_cast<float_X>(calcTWTSFieldX(pos, time_SI));
            else
            {
                if constexpr(T_component == 1)
                {
                    return static_cast<float_X>(calcTWTSFieldY(pos, time_SI));
                }
                if constexpr(T_component == 2)
                {
                    return static_cast<float_X>(calcTWTSFieldZ(pos, time_SI));
                }
            }

            // we should never be here
            return NAN;
        }
        if constexpr(simDim != DIM3)
            return (*this)(cellIdx, currentStep)[T_component];
    }

    template<typename T_Field>
    HDINLINE std::array<float_T, 11u> TWTSTight<T_Field>::defineBasicHelperVariables(
        float_64 const& phi,
        float_64 const& beta_0,
        float_64 const& wavelength_SI,
        float_64 const& pulselength_SI,
        float_64 const& w_x_SI)
    {
        /* If phi < 0 the formulas below are not directly applicable.
         * Instead phi is taken positive, but the entire pulse rotated by 180 deg around the
         * y-axis of the coordinate system in this function.
         */
        auto const absPhi = float_T(math::abs(phi));

        /* Propagation speed of overlap normalized to the speed of light [Default: beta0=1.0] */
        auto const beta0 = float_T(beta_0);

        float_T sinPhi;
        float_T cosPhi;
        math::sincos(absPhi, sinPhi, cosPhi);
        float_T const tanAlpha = (float_T(1.0) - beta0 * cosPhi) / (beta0 * sinPhi);

        auto const cspeed = float_T(sim.si.getSpeedOfLight() / sim.unit.speed());
        auto const lambda0 = float_T(wavelength_SI / sim.unit.length());
        float_T const omega0 = float_T(2.0 * PI) * cspeed / lambda0;
        /* factor 2  in tauG arises from definition convention in laser formula */
        auto const tauG = float_T(pulselength_SI * 2.0 / sim.unit.time());
        /* w0 is wx here --> w0 could be replaced by wx */
        auto const w0 = float_T(w_x_SI / sim.unit.length());
        auto const k = float_T(2.0 * PI / lambda0);

        return std::array<float_T, 11u>{absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k};
    }

    template<typename T_Field>
    HDINLINE bool TWTSTight<T_Field>::isOutsideTWTSEnvelope(std::array<float_T, 4u> const& minimalCoordinates) const
    {
        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        auto const [x, y, z, t] = minimalCoordinates;

        /* To avoid underflows in computation, fields are set to zero
         * before and after the respective TWTS pulse envelope.
         */
        if(math::abs(y - z * tanAlpha - (beta0 * cspeed * t)) > (numSigmas * tauG * cspeed))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    template<typename T_Field>
    HDINLINE std::array<float_T, 4> TWTSTight<T_Field>::defineMinimalCoordinates(
        float3_64 const& pos,
        float_64 const& time) const
    {
        /* In order to calculate in single-precision and in order to account for errors in
         * the approximations far from the coordinate origin, we use the wavelength-periodicity and
         * the known propagation direction for realizing the laser pulse using relative coordinates
         * (i.e. from a finite coordinate range) only. All these quantities have to be calculated
         * in double precision.
         */
        float_64 const deltaT
            = wavelength_SI / sim.si.getSpeedOfLight() / (1.0 - beta_0 * math::cos(precisionCast<float_64>(phi)));
        float_64 const deltaY = beta_0 * sim.si.getSpeedOfLight() * deltaT;
        float_64 const numberOfPeriods = math::floor(time / deltaT);
        auto const timeMod = float_T(time - numberOfPeriods * deltaT);
        auto const yMod = float_T(pos.y() - numberOfPeriods * deltaY);

        auto const x = float_T(phiPositive * pos.x() / sim.unit.length());
        auto const y = float_T(yMod / sim.unit.length());
        auto const z = float_T(phiPositive * pos.z() / sim.unit.length());
        auto const t = float_T(timeMod / sim.unit.time());

        return std::array<float_T, 4u>{x, y, z, t};
    }

    template<typename T_Field>
    HDINLINE std::array<float_T, 7u> TWTSTight<T_Field>::defineTrigonometryShortcuts(
        std::array<float_T, 11u> const& basicTWTSHelperVariables,
        float_T const& polAngle)
    {
        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        /* Calculating shortcuts for speeding up field calculation */
        float_T const tanPhi = math::tan(absPhi);
        float_T const cotPhi = float_T(1.0) / tanPhi;
        float_T const sinPhi_2 = sinPhi * sinPhi;
        float_T const cosPhi_2 = cosPhi * cosPhi;
        float_T const sinPolAngle = float_T(math::sin(polAngle));
        float_T const cosPolAngle = float_T(math::cos(polAngle));
        float_T const sin2Phi = math::sin(float_T(2.0) * absPhi);

        return std::array<float_T, 7u>{tanPhi, cotPhi, sinPhi_2, cosPhi_2, sinPolAngle, cosPolAngle, sin2Phi};
    }

    template<typename T_Field>
    HDINLINE std::tuple<std::array<float_T, 8u>, std::array<complex_T, 6u>> TWTSTight<
        T_Field>::defineCommonHelperVariables(std::array<float_T, 4u> const& minimalCoordinates) const
    {
        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        auto const [x, y, z, t] = minimalCoordinates;

        float_T const x2 = x * x;
        float_T const tauG2 = tauG * tauG;
        float_T const psi0 = float_T(2.0) / k;
        float_T const w02 = w0 * w0;
        float_T const beta02 = beta0 * beta0;
        float_T const nu = (y * cosPhi - z * sinPhi) / cspeed;
        float_T const xi = (-z * cosPhi - y * sinPhi) * tanAlpha / cspeed;
        float_T const besselI0const = math::bessel::i0(k * k * sinPhi * w02 / float_T(2.0));

        complex_T const Xm = -z - complex_T(0, 0.5) * (k * w02);
        complex_T const rhom = math::sqrt(x2 + math::cPow(Xm, static_cast<uint32_t>(2u)));
        complex_T const Xm2 = Xm * Xm;
        complex_T const rhom2 = rhom * rhom;
        complex_T const besselJ0const = math::bessel::j0(k * sinPhi * rhom);
        complex_T const besselJ1const = math::bessel::j1(k * sinPhi * rhom);

        return std::make_tuple(
            std::array<float_T, 8u>{x2, tauG2, psi0, w02, beta02, nu, xi, besselI0const},
            std::array<complex_T, 6u>{Xm, rhom, Xm2, rhom2, besselJ0const, besselJ1const});
    }

    template<typename T_Field>
    HDINLINE complex_T TWTSTight<T_Field>::defineTWTSEnvelope(
        std::array<float_T, 4u> const& minimalCoordinates,
        std::tuple<std::array<float_T, 8u>, std::array<complex_T, 6u>> const& commonHelperVariables) const
    {
        auto const [absPhi, sinPhi, cosPhi, beta0, tanAlpha, cspeed, lambda0, omega0, tauG, w0, k]
            = basicTWTSHelperVariables;
        auto const [x, y, z, t] = minimalCoordinates;
        auto const [tanPhi, cotPhi, sinPhi_2, cosPhi_2, sinPolAngle, cosPolAngle, sin2Phi] = trigonometryShortcuts;
        auto const [floatHelpers, complexHelpers] = defineCommonHelperVariables(minimalCoordinates);
        auto const [x2, tauG2, psi0, w02, beta02, nu, xi, besselI0const] = floatHelpers;

        complex_T const zeroOrder
            = (beta0 * tauG)
              / (math::sqrt(float_T(2.0))
                 * math::exp(
                     beta02 * omega0 * math::cPow(t - nu + xi, static_cast<uint32_t>(2u))
                     / (beta02 * omega0 * tauG2 - complex_T(0, 2) * (beta02 * (nu - xi) * cotPhi * cotPhi)
                        + complex_T(0, 2) * (beta0 * (float_T(2.0) * nu - xi) * cotPhi / sinPhi)
                        - complex_T(0, 2) * (nu / sinPhi_2)))
                 * math::sqrt(
                     ((beta02 * omega0 * tauG2) / float_T(2.0) - I * (beta02 * (nu - xi) * cotPhi * cotPhi)
                      + I * (beta0 * (float_T(2.0) * nu - xi) * cotPhi / sinPhi) - I * (nu / sinPhi_2))
                     / omega0));

        return zeroOrder;
    }
} /* namespace picongpu::templates::twtstight */
