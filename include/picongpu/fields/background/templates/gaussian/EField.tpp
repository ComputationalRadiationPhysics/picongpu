/* Copyright 2014-2020 Alexander Debus, Axel Huebl
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/math/Complex.hpp>

#include "picongpu/fields/background/templates/gaussian/RotateField.tpp"
#include "picongpu/fields/background/templates/gaussian/GetInitialTimeDelay_SI.tpp"
#include "picongpu/fields/background/templates/gaussian/getFieldPositions_SI.tpp"
#include "picongpu/fields/background/templates/gaussian/EField.hpp"
#include "picongpu/fields/CellType.hpp"

namespace picongpu
{
/* Load pre-defined background field */
namespace templates
{
/* Gaussian laser pulse */
namespace gaussian
{

    HINLINE
    EField::EField( const float_64 focus_y_SI,
                    const float_64 wavelength_SI,
                    const float_64 pulselength_SI,
                    const float_64 w0_SI,
                    const float_X phi,
                    const float_64 tdelay_user_SI,
                    const bool auto_tdelay,
                    const PolarizationType pol ) :
        focus_y_SI(focus_y_SI), wavelength_SI(wavelength_SI),
        pulselength_SI(pulselength_SI), w0_SI(w0_SI),
        phi(phi), tdelay_user_SI(tdelay_user_SI), dt(SI::DELTA_T_SI),
        unit_length(UNIT_LENGTH), auto_tdelay(auto_tdelay), pol(pol)
    {
        /* Note: Enviroment-objects cannot be instantiated on CUDA GPU device. Since this is done
                 on host (see fieldBackground.param), this is no problem.
         */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::getInitialTimeDelay_SI( auto_tdelay, tdelay_user_SI,
                                                 halfSimSize, pulselength_SI,
                                                 focus_y_SI, w0_SI, phi );
    }

    HDINLINE float3_X
    EField::getEfield_Normalized_Linear_X(
        const pmacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
        const float_64 time) const
    {
        typedef pmacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        PosVecVec pos(PosVecVec::create(
                                           float3_64::create(0.0)
                                       ));

        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i) pos[k][i] = eFieldPositions_SI[k][i];
        }

        /* An example of intra-cell position offsets is the staggered Yee-grid.
         *
         * Calculate Ey-component with the intra-cell offset of a Ey-field
         */
        const float_64 Ey_Ey = calcEy_Linear_X(pos[1], time);
        /* Calculate Ez-component with the intra-cell offset of a Ey-field */
        const float_64 Ez_Ey = calcEz_Linear_X(pos[1], time);
        /* Calculate Ey-component with the intra-cell offset of a Ez-field */
        const float_64 Ey_Ez = calcEy_Linear_X(pos[2], time);
        /* Calculate Ez-component with the intra-cell offset of a Ez-field */
        const float_64 Ez_Ez = calcEz_Linear_X(pos[2], time);

        /* Since we rotated all position vectors before calling calcEy,
         * we need to back-rotate the resulting E-field vector.
         *
         * RotationMatrix[phi-PI/2].(Ey,Ez) for rotating back the field-vectors.
         */
        const float_64 Ey_rot = +math::sin(+phi) * Ey_Ey + math::cos(+phi) * Ez_Ey;
        const float_64 Ez_rot = -math::cos(+phi) * Ey_Ez + math::sin(+phi) * Ez_Ez;

        /* Finally, the E-field normalized to the peak amplitude. */
        return float3_X( float_X( calcEx_Linear_X(pos[0], time) ),
                         float_X( Ey_rot ),
                         float_X( Ez_rot ) );
    }

    HDINLINE float3_X
    EField::getEfield_Normalized_Linear_YZ(
                const pmacc::math::Vector<floatD_64,detail::numComponents>& eFieldPositions_SI,
                const float_64 time) const
    {
        typedef pmacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        PosVecVec pos(PosVecVec::create(
                                           float3_64::create(0.0)
                                       ));

        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i) pos[k][i] = eFieldPositions_SI[k][i];
        }

        /* An example of intra-cell position offsets is the staggered Yee-grid.
         *
         * Calculate Ey-component with the intra-cell offset of a Ey-field
         */
        const float_64 Ey_Ey = calcEy_Linear_YZ(pos[1], time);
        /* Calculate Ez-component with the intra-cell offset of a Ey-field */
        const float_64 Ez_Ey = calcEz_Linear_YZ(pos[1], time);
        /* Calculate Ey-component with the intra-cell offset of a Ez-field */
        const float_64 Ey_Ez = calcEy_Linear_YZ(pos[2], time);
        /* Calculate Ez-component with the intra-cell offset of a Ez-field */
        const float_64 Ez_Ez = calcEz_Linear_YZ(pos[2], time);

        /* Since we rotated all position vectors before calling calcEy,
         * we need to back-rotate the resulting E-field vector.
         *
         * RotationMatrix[phi-PI/2].(Ey,Ez) for rotating back the field-vectors.
         */
        const float_64 Ey_rot = +math::sin(+phi) * Ey_Ey + math::cos(+phi) * Ez_Ey;
        const float_64 Ez_rot = -math::cos(+phi) * Ey_Ez + math::sin(+phi) * Ez_Ez;

        /* Finally, the E-field normalized to the peak amplitude. */
        return float3_X( float_X( calcEx_Linear_YZ(pos[0], time) ),
                         float_X( Ey_rot ),
                         float_X( Ez_rot ) );
    }

    HDINLINE float3_X
    EField::operator()( const DataSpace<simDim>& cellIdx,
                        const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * dt - tdelay;
        const traits::FieldPosition<fields::CellType, FieldE> fieldPosE;

        const pmacc::math::Vector<floatD_64,detail::numComponents> eFieldPositions_SI =
              detail::getFieldPositions_SI(cellIdx, halfSimSize,
                fieldPosE(), unit_length, focus_y_SI, phi);

        /* Single Gaussian laser pulse */
        switch (pol)
        {
            case LINEAR_X :
            return getEfield_Normalized_Linear_X(eFieldPositions_SI, time_SI);

            case LINEAR_YZ :
            return getEfield_Normalized_Linear_YZ(eFieldPositions_SI, time_SI);
        }
        return getEfield_Normalized_Linear_X(eFieldPositions_SI, time_SI); // defensive default
    }

    /** Calculate the Ex_Linear_X(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcEx_Linear_X( const float3_64& pos, const float_64 time) const
    {
        using complex_T = pmacc::math::Complex< float_T >;
        using complex_64 = pmacc::math::Complex< float_64 >;
        /* Unit of speed */
        const float_64 UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /* Unit of time */
        const float_64 UNIT_TIME = SI::DELTA_T_SI;
        /* Unit of length */
        const float_64 UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;

        const float_T cspeed = float_T( SI::SPEED_OF_LIGHT_SI / UNIT_SPEED );
        const float_T lambda0 = float_T( wavelength_SI / UNIT_LENGTH );
        const float_T omega0 = float_T( 2.0 * PI * cspeed / lambda0 );
        /* factor 2 in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T( pulselength_SI * 2.0 / UNIT_TIME );
        const float_T w0 = float_T( w0_SI / UNIT_LENGTH );
        const float_T Z0 = float_T( PI * w0 * w0 / lambda0 );
        const float_T k0 = float_T( 2.0 * PI / lambda0 );
        const float_T x = float_T( pos.x() / UNIT_LENGTH );
        const float_T y = float_T( pos.y() / UNIT_LENGTH );
        const float_T z = float_T( pos.z() / UNIT_LENGTH );
        const float_T t = float_T( time / UNIT_TIME );

        const float_T wz = math::sqrt( w0 * w0 * ( float_T( 1.0 ) + z * z / ( Z0 * Z0 ) ) );
        /* Following line can produce NaNs, but we only need the inverse:
         * const float_T Rz = z * ( float_T( 1.0 ) + ( Z0 * Z0 ) / ( z * z ) );
         */
        const float_T RzInv = z / ( z * z +  Z0 * Z0 );
        const float_T phiz = math::atan( z / Z0 );
        const float_T phi0 = float_T( 0.0 ); /* carrier envelope phase ( prepared extension ) */
        const float_T xi = x / w0;
        const float_T eta = y / w0;
        const float_T zeta = z / ( k0 * w0 * w0 );
        const float_T rho2 = xi * xi + eta * eta;
        const complex_T theta = complex_T( 1, 0 ) / ( complex_T( 0, 1 ) + float_T( 2.0 ) * zeta );
        const float_T sigma = t / tauG;
        const float_T epsilon = float_T( 1.0 ) / ( omega0 * tauG );
        /* Not used, but left here to facilitate later refactoring:
         * const float_T s = float_T( 1.0 ) / ( k0 * w0 );
         * */

        /* Note: The E-field amplitude E_0 is multiplied later outside this function */
        const complex_T Ex = w0 / wz * math::exp(
                             complex_T( 0, 1 ) * ( omega0 * t - k0 * z )
                           + complex_T( 0, 1 ) * phiz + complex_T( 0, 1 ) * phi0
                           - ( x * x + y * y) *
                               ( float_T( 1.0 ) / ( wz * wz )
                                 + complex_T( 0, 1 ) * k0 * RzInv / float_T( 2.0 ) )
                          );
        const complex_T Ex_0 = Ex * math::exp( -(t - z / cspeed) * (t - z / cspeed)
                               / ( float_T( 2.0 ) * tauG * tauG ) );
        const complex_T Ex_1 = Ex_0 * ( float_T( 1.0 ) + sigma * epsilon * theta
                                        * ( -float_T( 2.0 ) * zeta * complex_T( 0, 1 )
                                            +float_T( 2.0 ) * rho2 -float_T( 2.0 ) * rho2 * theta * zeta ) );

        const complex_T result = Ex_1;
        return result.get_real();
        /* TO DO: Identify places where complex_64 is necessary (phase, complex divisions etc.). */
    }

    /** Calculate the Ey_Linear_X(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcEy_Linear_X( const float3_64& pos, const float_64 time) const
    {
        return float_T( 0.0 );
    }

    /** Calculate the Ez_Linear_X(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcEz_Linear_X( const float3_64& pos, const float_64 time) const
    {
        using complex_T = pmacc::math::Complex< float_T >;
        using complex_64 = pmacc::math::Complex< float_64 >;
        /* Unit of speed */
        const float_64 UNIT_SPEED = SI::SPEED_OF_LIGHT_SI;
        /* Unit of time */
        const float_64 UNIT_TIME = SI::DELTA_T_SI;
        /* Unit of length */
        const float_64 UNIT_LENGTH = UNIT_TIME*UNIT_SPEED;

        const float_T cspeed = float_T( SI::SPEED_OF_LIGHT_SI / UNIT_SPEED );
        const float_T lambda0 = float_T( wavelength_SI / UNIT_LENGTH );
        const float_T omega0 = float_T( 2.0 * PI * cspeed / lambda0 );
        /* factor 2 in tauG arises from definition convention in laser formula */
        const float_T tauG = float_T( pulselength_SI * 2.0 / UNIT_TIME );
        const float_T w0 = float_T( w0_SI / UNIT_LENGTH );
        const float_T Z0 = float_T( PI * w0 * w0 / lambda0 );
        const float_T k0 = float_T( 2.0 * PI / lambda0 );
        const float_T x = float_T( pos.x() / UNIT_LENGTH );
        const float_T y = float_T( pos.y() / UNIT_LENGTH );
        const float_T z = float_T( pos.z() / UNIT_LENGTH );
        const float_T t = float_T( time / UNIT_TIME );

        const float_T wz = math::sqrt( w0 * w0 * ( float_T( 1.0 ) + z * z / ( Z0 * Z0 ) ) );
         /* Following line can produce NaNs, but we only need the inverse:
         * const float_T Rz = z * ( float_T( 1.0 ) + ( Z0 * Z0 ) / ( z * z ) );
         */
        const float_T RzInv = z / ( z * z +  Z0 * Z0 );
        const float_T phiz = math::atan( z / Z0 );
        const float_T phi0 = float_T( 0.0 ); /* carrier envelope phase ( prepared extension ) */
        const float_T xi = x / w0;
        const float_T eta = y / w0;
        const float_T zeta = z / ( k0 * w0 * w0 );
        const float_T rho2 = xi * xi + eta * eta;
        const complex_T theta = complex_T( 1, 0 ) / ( complex_T( 0, 1 ) + float_T( 2.0 ) * zeta );
        const float_T sigma = t / tauG;
        const float_T epsilon = float_T( 1.0 ) / ( omega0 * tauG );
        const float_T s = float_T( 1.0 ) / ( k0 * w0 );

        /* Note: The E-field amplitude E_0 is multiplied later outside this function */
        const complex_T Ex = w0 / wz * math::exp(
                             complex_T( 0, 1 ) * ( omega0 * t - k0 * z )
                           + complex_T( 0, 1 ) * phiz + complex_T( 0, 1 ) * phi0
                           - ( x * x + y * y) *
                               ( float_T( 1.0 ) / ( wz * wz )
                                 + complex_T( 0, 1 ) * k0 * RzInv / float_T( 2.0 ) )
                          );
        const complex_T Ex_0 = Ex * math::exp( -(t - z / cspeed) * (t - z / cspeed)
                               / ( float_T( 2.0 ) * tauG * tauG ) );
        const complex_T Ez_1 = s * xi * ( -float_T( 2.0 ) * theta ) * Ex_0 *
                                   ( float_T( 1.0 ) + sigma * epsilon * theta *
                                     (   - float_T( 2.0 ) * zeta * complex_T( 0, 1 )
                                         + float_T( 2.0 ) * rho2
                                         - float_T( 2.0 ) * rho2 * theta * zeta - float_T( 1.0 )
                                     )
                                   );

        const complex_T result = Ez_1;
        return result.get_real();
        /* TO DO: Identify places where complex_64 is necessary (phase, complex divisions etc.). */
    }

    /** Calculate the Ex_Linear_YZ(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcEx_Linear_YZ( const float3_64& pos, const float_64 time) const
    {
        /* The field function of Ex (polarization in YZ-plane)
         * is by definition identical to -Ey (for linear polarization in x)
         */
        return -calcEy_Linear_X( pos, time );
    }

    /** Calculate the Ey_Linear_YZ(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcEy_Linear_YZ( const float3_64& pos, const float_64 time) const
    {
        /* The field function of Ey (polarization in YZ-plane)
         * is by definition identical to Ex (for linear polarization in x)
         */
        return calcEx_Linear_X( pos, time );
    }

    /** Calculate the Ez_Linear_YZ(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations) for calculating
     *             the field */
    HDINLINE EField::float_T
    EField::calcEz_Linear_YZ( const float3_64& pos, const float_64 time) const
    {
        /* The field function of Ez (polarization in pulse-front-tilt plane)
         * is by definition identical to Ez (for linear polarizations in x)
         */
        return calcEz_Linear_X( pos, time );
    }

} /* namespace gaussian */
} /* namespace templates */
} /* namespace picongpu */
