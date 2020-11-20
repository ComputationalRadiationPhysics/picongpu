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
#include "picongpu/fields/background/templates/gaussian/BField.hpp"
#include "picongpu/fields/CellType.hpp"


namespace picongpu
{
/** Load pre-defined background field */
namespace templates
{
/** Gaussian laser pulse */
namespace gaussian
{

    HINLINE
    BField::BField( const float_64 focus_y_SI,
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
         * on host (see fieldBackground.param), this is no problem.
         */
        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        halfSimSize = subGrid.getGlobalDomain().size / 2;
        tdelay = detail::getInitialTimeDelay_SI( auto_tdelay, tdelay_user_SI,
                                                 halfSimSize, pulselength_SI,
                                                 focus_y_SI, w0_SI, phi );
    }

    HDINLINE float3_X
    BField::getBfield_Normalized_Linear_X(
            const pmacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        typedef pmacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        PosVecVec pos(PosVecVec::create(
                                           float3_64::create(0.0)
                                       ));

        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i)
                pos[k][i] = bFieldPositions_SI[k][i];
        }

        /* An example of intra-cell position offsets is the staggered Yee-grid.
         *
         * Calculate By-component with the intra-cell offset of a By-field
         */
        const float_64 By_By = calcBy_Linear_X(pos[1], time);
        /* Calculate Bz-component the the intra-cell offset of a By-field */
        const float_64 Bz_By = calcBz_Linear_X(pos[1], time);
        /* Calculate By-component the the intra-cell offset of a Bz-field */
        const float_64 By_Bz = calcBy_Linear_X(pos[2], time);
        /* Calculate Bz-component the the intra-cell offset of a Bz-field */
        const float_64 Bz_Bz = calcBz_Linear_X(pos[2], time);
        /* Since we rotated all position vectors before calling calcBy and calcBz_Ex,
         * we need to back-rotate the resulting B-field vector.
         *
         * RotationMatrix[phi-PI/2].(By,Bz) for rotating back the field vectors.
         */
        const float_64 By_rot = +math::sin(+phi) * By_By + math::cos(+phi) * Bz_By;
        const float_64 Bz_rot = -math::cos(+phi) * By_Bz + math::sin(+phi) * Bz_Bz;

        /* Finally, the B-field normalized to the peak amplitude.
         * Calculate Bx-component with the intra-cell offset of a Bx-field
         */
        return float3_X( float_X( calcBx_Linear_X(pos[0], time) ),
                         float_X( By_rot ),
                         float_X( Bz_rot ) );
    }

    HDINLINE float3_X
    BField::getBfield_Normalized_Linear_YZ(
            const pmacc::math::Vector<floatD_64,detail::numComponents>& bFieldPositions_SI,
            const float_64 time) const
    {
        typedef pmacc::math::Vector<float3_64,detail::numComponents> PosVecVec;
        PosVecVec pos(PosVecVec::create(
                                           float3_64::create(0.0)
                                       ));

        for (uint32_t k = 0; k<detail::numComponents;++k) {
            for (uint32_t i = 0; i<simDim;++i) pos[k][i] = bFieldPositions_SI[k][i];
        }

        /* An example of intra-cell position offsets is the staggered Yee-grid.
         *
         * Calculate By-component with the intra-cell offset of a By-field
         */
        const float_64 By_By = calcBy_Linear_YZ(pos[1], time);
        /* Calculate Bz-component the the intra-cell offset of a By-field */
        const float_64 Bz_By = calcBz_Linear_YZ(pos[1], time);
        /* Calculate By-component the the intra-cell offset of a Bz-field */
        const float_64 By_Bz = calcBy_Linear_YZ(pos[2], time);
        /* Calculate Bz-component the the intra-cell offset of a Bz-field */
        const float_64 Bz_Bz = calcBz_Linear_YZ(pos[2], time);
        /* Since we rotated all position vectors before calling calcBy and calcBz_Ex,
         * we need to back-rotate the resulting B-field vector.
         *
         * RotationMatrix[phi-PI/2].(By,Bz) for rotating back the field vectors.
         */
        const float_64 By_rot = +math::sin(+phi) * By_By + math::cos(+phi) * Bz_By;
        const float_64 Bz_rot = -math::cos(+phi) * By_Bz + math::sin(+phi) * Bz_Bz;

        /* Finally, the B-field normalized to the peak amplitude. */
        return float3_X( float_X( calcBx_Linear_YZ(pos[0], time) ),
                         float_X( By_rot ),
                         float_X( Bz_rot ) );
    }

    HDINLINE float3_X
    BField::operator()( const DataSpace<simDim>& cellIdx,
                            const uint32_t currentStep ) const
    {
        const float_64 time_SI = float_64(currentStep) * dt - tdelay;
        const traits::FieldPosition<fields::CellType, FieldB> fieldPosB;

        const pmacc::math::Vector<floatD_64,detail::numComponents> bFieldPositions_SI =
              detail::getFieldPositions_SI(cellIdx, halfSimSize,
                fieldPosB(), unit_length, focus_y_SI, phi);
        /* Single Gaussian laser pulse */
        switch (pol)
        {
            case LINEAR_X :
            return getBfield_Normalized_Linear_X(bFieldPositions_SI, time_SI);

            case LINEAR_YZ :
            return getBfield_Normalized_Linear_YZ(bFieldPositions_SI, time_SI);
        }
        return getBfield_Normalized_Linear_X(bFieldPositions_SI, time_SI); // defensive default
    }

    /** Calculate the Bx_Linear_X(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcBx_Linear_X( const float3_64& pos, const float_64 time ) const
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
        /* factor sqrt(2) in tauG arises from the convention to parameterize
         * the standard deviation of intensity instead of the field.
         * */
        const float_T tauG = float_T( pulselength_SI * math::sqrt( float_T( 2.0 ) ) / UNIT_TIME );
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
        const complex_T Bx_1 = s * s * xi * eta * ( -float_T( 4.0 ) * theta * theta )
                               * Ex_0 * ( float_T( 1.0 ) + sigma * epsilon * theta
                                        * ( -float_T( 2.0 ) * zeta * complex_T( 0, 1 )
                                            +float_T( 2.0 ) * rho2 - float_T( 2.0 ) * rho2 * theta * zeta - float_T( 2.0 ) )
                                        );
        const complex_T result = Bx_1 / cspeed;
        return result.get_real() / UNIT_SPEED;
        /* TO DO: Identify places where complex_64 is necessary (phase, complex divisions etc.). */
    }

    /** Calculate the By_Linear_X(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcBy_Linear_X( const float3_64& pos, const float_64 time ) const
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
        /* factor sqrt(2) in tauG arises from the convention to parameterize
         * the standard deviation of intensity instead of the field.
         * */
        const float_T tauG = float_T( pulselength_SI * math::sqrt( float_T( 2.0 ) ) / UNIT_TIME );
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
        const complex_T By_1 = Ex_0 * ( float_T( 1.0 ) + sigma * epsilon * theta
                                        * ( -float_T( 2.0 ) * zeta * complex_T( 0, 1 )
                                            +float_T( 2.0 ) * rho2 - float_T( 2.0 ) * rho2 * theta * zeta )
                                        + s * s * ( -float_T( 2.0 ) * rho2 * theta * theta
                                                    + float_T( 4.0 ) * theta * theta * xi * xi )
                                                * ( float_T( 1.0 ) + sigma * epsilon * theta
                                        * ( -float_T( 2.0 ) * zeta * complex_T( 0, 1 )
                                            +float_T( 2.0 ) * rho2 - float_T( 2.0 ) * rho2 * theta * zeta - float_T( 2.0 ) )
                                                  )
                                      );
        const complex_T result = By_1 / cspeed;
        return result.get_real() / UNIT_SPEED;
        /* TO DO: Identify places where complex_64 is necessary (phase, complex divisions etc.). */
    }

    /** Calculate the Bz_Linear_X(r,t) field here
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcBz_Linear_X( const float3_64& pos, const float_64 time ) const
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
        /* factor sqrt(2) in tauG arises from the convention to parameterize
         * the standard deviation of intensity instead of the field.
         * */
        const float_T tauG = float_T( pulselength_SI * math::sqrt( float_T( 2.0 ) ) / UNIT_TIME );
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
        const complex_T Bz_1 = s * eta * ( -float_T( 2.0 ) * theta ) * Ex_0
                                       * ( float_T( 1.0 ) + sigma * epsilon * theta
                                           * ( -float_T( 2.0 ) * zeta * complex_T( 0, 1 )
                                               +float_T( 2.0 ) * rho2 - float_T( 2.0 ) * rho2 * theta * zeta - float_T( 1.0 )
                                             )
                                         );
        const complex_T result = Bz_1 / cspeed;
        return result.get_real() / UNIT_SPEED;
        /* TO DO: Identify places where complex_64 is necessary (phase, complex divisions etc.). */
    }

    /** Calculate the Bx_Linear_YZ(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcBx_Linear_YZ( const float3_64& pos, const float_64 time ) const
    {
        /* The Bx-field for the laser field polarized in the YZ-plane is the same as
         * the By-field for the laser field polarized in x except for the sign.
         */
        return -calcBy_Linear_X( pos, time );
    }

    /** Calculate the By_Linear_YZ(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcBy_Linear_YZ( const float3_64& pos, const float_64 time ) const
    {
        /* The By-field for the laser field polarized in the YZ-plane is the same as
         * the Bx-field for the laser field polarized in x.
         */
        return calcBx_Linear_X( pos, time );
    }

    /** Calculate the Bz_Linear_YZ(r,t) field
     *
     * \param pos Spatial position of the target field.
     * \param time Absolute time (SI, including all offsets and transformations)
     *             for calculating the field */
    HDINLINE BField::float_T
    BField::calcBz_Linear_YZ( const float3_64& pos, const float_64 time ) const
    {
        /* The Bz-field for the laser field polarized in the YZ-plane is the same as
         * the Bz-field for the laser field polarized in x.
         */
        return calcBz_Linear_YZ( pos, time );
    }

} /* namespace gaussian */
} /* namespace templates */
} /* namespace picongpu */
