/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe
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

#include "types.h"
#include "math/Vector.hpp"

namespace picongpu
{
    namespace leheSolver
    {
        using namespace PMacc;


        template< class Direction >
        struct CurlELehe;


        template< >
        struct CurlELehe< ::picongpu::fieldSolverLehe::CherenkovFreeDirection_X >
        {
            typedef PMacc::math::CT::Int< 1, 1, 1 > LowerMargin;
            typedef PMacc::math::CT::Int< 2, 2, 2 > UpperMargin;

            template<class Memory >
            HDINLINE typename Memory::ValueType operator( )(const Memory & mem ) const
            {
                /* Distinguished direction where the numerical Cherenkov Radiation
                 * of moving particles is suppressed.
                 */
                const float_X isDir_x = float_X( 1.0 );
                const float_X isDir_y = float_X( 0.0 );
                const float_X isDir_z = float_X( 0.0 );

                const float_X isNotDir_x = float_X( 1.0 ) - isDir_x;
                const float_X isNotDir_y = float_X( 1.0 ) - isDir_y;
                const float_X isNotDir_z = float_X( 1.0 ) - isDir_z;

                const float_X dx2 = CELL_WIDTH * CELL_WIDTH;
                const float_X dy2 = CELL_HEIGHT * CELL_HEIGHT;
                const float_X dz2 = CELL_DEPTH * CELL_DEPTH;
                const float_X dt2 = DELTA_T * DELTA_T;
                const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

                const float_X reci_dx = float_X( 1.0 ) / CELL_WIDTH;
                const float_X reci_dy = float_X( 1.0 ) / CELL_HEIGHT;
                const float_X reci_dz = float_X( 1.0 ) / CELL_DEPTH;

                const float_X beta_xy = float_X( 0.125 ) * dx2 / dy2 * isDir_x
                    + float_X( 0.125 ) * isNotDir_x * isDir_y;
                const float_X beta_xz = float_X( 0.125 ) * dx2 / dz2 * isDir_x
                    + float_X( 0.125 ) * isNotDir_x * isDir_z;

                const float_X beta_yx = float_X( 0.125 ) * dy2 / dx2 * isDir_y
                    + float_X( 0.125 ) * isNotDir_y * isDir_x;
                const float_X beta_yz = float_X( 0.125 ) * dy2 / dz2 * isDir_y
                    + float_X( 0.125 ) * isNotDir_y * isDir_z;

                const float_X beta_zx = float_X( 0.125 ) * dz2 / dx2 * isDir_z
                    + float_X( 0.125 ) * isNotDir_z * isDir_x;
                const float_X beta_zy = float_X( 0.125 ) * dz2 / dy2 * isDir_z
                    + float_X( 0.125 ) * isNotDir_z * isDir_y;

                const float_X d_dir = CELL_WIDTH * isDir_x
                    + CELL_HEIGHT * isDir_y
                    + CELL_DEPTH * isDir_z;
                const float_X d_dir2 = d_dir * d_dir;

                //! \todo all constants, calculate once at CPU and use as parameter
                const float_X mySin = math::sin( float_X( 0.5 ) * float_X( M_PI ) *
                                                 SPEED_OF_LIGHT * DELTA_T / d_dir );

                // delta_x0 == delta_x
                // delta_dir0 == delta_dir
                const float_X delta_dir0 = float_X( 0.25 ) *
                    ( float_X( 1.0 ) - d_dir2 / ( c2 * dt2 ) * mySin * mySin );

                const float_X alpha_x = float_X( 1.0 )
                    - float_X( 2.0 ) * beta_xy * isNotDir_x * isDir_y
                    - float_X( 2.0 ) * beta_xz * isNotDir_x * isDir_z
                    - float_X( 2.0 ) * beta_xy * isDir_x
                    - float_X( 2.0 ) * beta_xz * isDir_x
                    - float_X( 3.0 ) * delta_dir0 * isDir_x;

                const float_X alpha_y = float_X( 1.0 )
                    - float_X( 2.0 ) * beta_yx * isNotDir_y * isDir_x
                    - float_X( 2.0 ) * beta_yz * isNotDir_y * isDir_z
                    - float_X( 2.0 ) * beta_yx * isDir_y
                    - float_X( 2.0 ) * beta_yz * isDir_y
                    - float_X( 3.0 ) * delta_dir0 * isDir_y;

                const float_X alpha_z = float_X( 1.0 )
                    - float_X( 2.0 ) * beta_zx * isNotDir_z * isDir_x
                    - float_X( 2.0 ) * beta_zy * isNotDir_z * isDir_y
                    - float_X( 2.0 ) * beta_zx * isDir_z
                    - float_X( 2.0 ) * beta_zy * isDir_z
                    - float_X( 3.0 ) * delta_dir0 * isDir_z;


                const float_X curl_x
                    = (
                        alpha_y * ( mem[0][0][0].z( ) - mem[0][-1][0].z( ) )
                        + beta_yx * ( mem[1][0][0].z( ) - mem[1][-1][0].z( ) )
                        + beta_yx * ( mem[-1][0][0].z( ) - mem[-1][-1][0].z( ) )
                        ) * reci_dy
                    - (
                        alpha_z * ( mem[0][0][0].y( ) - mem[0][0][-1].y( ) )
                        + beta_zx * ( mem[1][0][0].y( ) - mem[1][0][-1].y( ) )
                        + beta_zx * ( mem[-1][0][0].y( ) - mem[-1][0][-1].y( ) )
                        ) * reci_dz;


                const float_X curl_y
                    = (
                        alpha_z * ( mem[0][0][0].x( ) - mem[0][0][-1].x( ) )
                        + beta_zx * ( mem[1][0][0].x( ) - mem[1][0][-1].x( ) )
                        + beta_zx * ( mem[-1][0][0].x( ) - mem[-1][0][-1].x( ) )
                        ) * reci_dz
                    - (
                        alpha_x * ( mem[0][0][0].z( ) - mem[-1][0][0].z( ) )
                        + delta_dir0 * ( mem[1][0][0].z( ) - mem[-2][0][0].z( ) )
                        + beta_xy * ( mem[0][1][0].z( ) - mem[-1][1][0].z( ) )
                        + beta_xy * ( mem[0][-1][0].z( ) - mem[-1][-1][0].z( ) )
                        + beta_xz * ( mem[0][0][1].z( ) - mem[-1][0][1].z( ) )
                        + beta_xz * ( mem[0][0][-1].z( ) - mem[-1][0][-1].z( ) )
                        ) * reci_dx;


                const float_X curl_z
                    = (
                        alpha_x * ( mem[0][0][0].y( ) - mem[-1][0][0].y( ) )
                        + delta_dir0 * ( mem[1][0][0].y( ) - mem[-2][0][0].y( ) )
                        + beta_xy * ( mem[0][1][0].y( ) - mem[-1][1][0].y( ) )
                        + beta_xy * ( mem[0][-1][0].y( ) - mem[-1][-1][0].y( ) )
                        + beta_xz * ( mem[0][0][1].y( ) - mem[-1][0][1].y( ) )
                        + beta_xz * ( mem[0][0][-1].y( ) - mem[-1][0][-1].y( ) )
                        ) * reci_dx
                    - (
                        alpha_y * ( mem[0][0][0].x( ) - mem[0][-1][0].x( ) )
                        + beta_yx * ( mem[1][0][0].x( ) - mem[1][-1][0].x( ) )
                        + beta_yx * ( mem[-1][0][0].x( ) - mem[-1][-1][0].x( ) )
                        ) * reci_dy;

                return float3_X( curl_x, curl_y, curl_z );

                //return float3_X(diff(mem, 1).z() - diff(mem, 2).y(),
                //                diff(mem, 2).x() - diff(mem, 0).z(),
                //                diff(mem, 0).y() - diff(mem, 1).x());
            }
        };


        template< >
        struct CurlELehe< ::picongpu::fieldSolverLehe::CherenkovFreeDirection_Y >
        {
            typedef PMacc::math::CT::Int< 1, 1, 1 > LowerMargin;
            typedef PMacc::math::CT::Int< 2, 2, 2 > UpperMargin;

            template<class Memory >
            HDINLINE typename Memory::ValueType operator( )(const Memory & mem ) const
            {
                /* Distinguished direction where the numerical Cherenkov Radiation
                 * of moving particles is suppressed.
                 */
                const float_X isDir_x = float_X( 0.0 );
                const float_X isDir_y = float_X( 1.0 );
                const float_X isDir_z = float_X( 0.0 );

                const float_X isNotDir_x = float_X( 1.0 ) - isDir_x;
                const float_X isNotDir_y = float_X( 1.0 ) - isDir_y;
                const float_X isNotDir_z = float_X( 1.0 ) - isDir_z;

                const float_X dx2 = CELL_WIDTH * CELL_WIDTH;
                const float_X dy2 = CELL_HEIGHT * CELL_HEIGHT;
                const float_X dz2 = CELL_DEPTH * CELL_DEPTH;
                const float_X dt2 = DELTA_T * DELTA_T;
                const float_X c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

                const float_X reci_dx = float_X( 1.0 ) / CELL_WIDTH;
                const float_X reci_dy = float_X( 1.0 ) / CELL_HEIGHT;
                const float_X reci_dz = float_X( 1.0 ) / CELL_DEPTH;

                /** Naming of the coefficients
                 *  1st letter: direction of differentiation
                 *  2nd letter: direction of averaging
                 */
                const float_X beta_xy = float_X( 0.125 ) * dx2 / dy2 * isDir_x
                    + float_X( 0.125 ) * isNotDir_x * isDir_y;
                const float_X beta_xz = float_X( 0.125 ) * dx2 / dz2 * isDir_x
                    + float_X( 0.125 ) * isNotDir_x * isDir_z;

                const float_X beta_yx = float_X( 0.125 ) * dy2 / dx2 * isDir_y
                    + float_X( 0.125 ) * isNotDir_y * isDir_x;
                const float_X beta_yz = float_X( 0.125 ) * dy2 / dz2 * isDir_y
                    + float_X( 0.125 ) * isNotDir_y * isDir_z;

                const float_X beta_zx = float_X( 0.125 ) * dz2 / dx2 * isDir_z
                    + float_X( 0.125 ) * isNotDir_z * isDir_x;
                const float_X beta_zy = float_X( 0.125 ) * dz2 / dy2 * isDir_z
                    + float_X( 0.125 ) * isNotDir_z * isDir_y;

                const float_X d_dir = CELL_WIDTH * isDir_x
                    + CELL_HEIGHT * isDir_y
                    + CELL_DEPTH * isDir_z;
                const float_X d_dir2 = d_dir * d_dir;

                /** \todo all constants, calculate once at CPU and use as parameter
                 *  \todo cast to float_64 to force high-precision sinus
                 */
                const float_X mySin = math::sin( float_X( 0.5 ) * float_X( M_PI ) *
                                                 SPEED_OF_LIGHT * DELTA_T / d_dir );

                // delta_y0 == delta_y
                // delta_dir0 == delta_dir
                const float_X delta_dir0 = float_X( 0.25 ) *
                    ( float_X( 1.0 ) - d_dir2 / ( c2 * dt2 ) * mySin * mySin );

                const float_X alpha_x = float_X( 1.0 )
                    - float_X( 2.0 ) * beta_xy * isNotDir_x * isDir_y
                    - float_X( 2.0 ) * beta_xz * isNotDir_x * isDir_z
                    - float_X( 2.0 ) * beta_xy * isDir_x
                    - float_X( 2.0 ) * beta_xz * isDir_x
                    - float_X( 3.0 ) * delta_dir0 * isDir_x;

                const float_X alpha_y = float_X( 1.0 )
                    - float_X( 2.0 ) * beta_yx * isNotDir_y * isDir_x
                    - float_X( 2.0 ) * beta_yz * isNotDir_y * isDir_z
                    - float_X( 2.0 ) * beta_yx * isDir_y
                    - float_X( 2.0 ) * beta_yz * isDir_y
                    - float_X( 3.0 ) * delta_dir0 * isDir_y;

                const float_X alpha_z = float_X( 1.0 )
                    - float_X( 2.0 ) * beta_zx * isNotDir_z * isDir_x
                    - float_X( 2.0 ) * beta_zy * isNotDir_z * isDir_y
                    - float_X( 2.0 ) * beta_zx * isDir_z
                    - float_X( 2.0 ) * beta_zy * isDir_z
                    - float_X( 3.0 ) * delta_dir0 * isDir_z;

                // Typedef an accessor to access mem[z][y][x]
                // in (x,y,z) order :)
                typedef DataSpace<DIM3> Space;

                const float_X curl_x
                    = (
                        alpha_y * ( mem(Space(0,0,0)*(-1)).z( ) - mem(Space(0,-1,0)*(-1)).z( ) )
                        + beta_yz * ( mem(Space(0,0,1)*(-1)).z( ) - mem(Space(0,-1,1)*(-1)).z( ) )
                        + beta_yz * ( mem(Space(0,0,-1)*(-1)).z( ) - mem(Space(0,-1,-1)*(-1)).z( ) )
                        + beta_yx * ( mem(Space(1,0,0)*(-1)).z( ) - mem(Space(1,-1,0)*(-1)).z( ) )
                        + beta_yx * ( mem(Space(-1,0,0)*(-1)).z( ) - mem(Space(-1,-1,0)*(-1)).z( ) )
                        + delta_dir0 * ( mem(Space(0,1,0)*(-1)).z( ) - mem(Space(0,-2,0)*(-1)).z( ) )
                        ) * reci_dy
                    - (
                        alpha_z * ( mem(Space(0,0,0)*(-1)).y( ) - mem(Space(0,0,-1)*(-1)).y( ) )
                        + beta_zx * ( mem(Space(1,0,0)*(-1)).y( ) - mem(Space(1,0,-1)*(-1)).y( ) )
                        + beta_zx * ( mem(Space(-1,0,0)*(-1)).y( ) - mem(Space(-1,0,-1)*(-1)).y( ) )
                        + beta_zy * ( mem(Space(0,1,0)*(-1)).y( ) - mem(Space(0,1,-1)*(-1)).y( ) )
                        + beta_zy * ( mem(Space(0,-1,0)*(-1)).y( ) - mem(Space(0,-1,-1)*(-1)).y( ) )
                        ) * reci_dz;


                const float_X curl_y
                    = (
                        alpha_z * ( mem(Space(0,0,0)*(-1)).x( ) - mem(Space(0,0,-1)*(-1)).x( ) )
                        + beta_zx * ( mem(Space(1,0,0)*(-1)).x( ) - mem(Space(1,0,-1)*(-1)).x( ) )
                        + beta_zx * ( mem(Space(-1,0,0)*(-1)).x( ) - mem(Space(-1,0,-1)*(-1)).x( ) )
                        + beta_zy * ( mem(Space(0,1,0)*(-1)).x( ) - mem(Space(0,1,-1)*(-1)).x( ) )
                        + beta_zy * ( mem(Space(0,-1,0)*(-1)).x( ) - mem(Space(0,-1,-1)*(-1)).x( ) )
                        ) * reci_dz
                    - (
                        alpha_x * ( mem(Space(0,0,0)*(-1)).z( ) - mem(Space(-1,0,0)*(-1)).z( ) )
                        + beta_xy * ( mem(Space(0,1,0)*(-1)).z( ) - mem(Space(-1,1,0)*(-1)).z( ) )
                        + beta_xy * ( mem(Space(0,-1,0)*(-1)).z( ) - mem(Space(-1,-1,0)*(-1)).z( ) )
                        + beta_xz * ( mem(Space(0,0,1)*(-1)).z( ) - mem(Space(-1,0,1)*(-1)).z( ) )
                        + beta_xz * ( mem(Space(0,0,-1)*(-1)).z( ) - mem(Space(-1,0,-1)*(-1)).z( ) )
                        ) * reci_dx;


                const float_X curl_z
                    = (
                        alpha_x * ( mem(Space(0,0,0)*(-1)).y( ) - mem(Space(-1,0,0)*(-1)).y( ) )
                        + beta_xy * ( mem(Space(0,1,0)*(-1)).y( ) - mem(Space(-1,1,0)*(-1)).y( ) )
                        + beta_xy * ( mem(Space(0,-1,0)*(-1)).y( ) - mem(Space(-1,-1,0)*(-1)).y( ) )
                        + beta_xz * ( mem(Space(0,0,1)*(-1)).y( ) - mem(Space(-1,0,1)*(-1)).y( ) )
                        + beta_xz * ( mem(Space(0,0,-1)*(-1)).y( ) - mem(Space(-1,0,-1)*(-1)).y( ) )
                        ) * reci_dx
                    - (
                        alpha_y * ( mem(Space(0,0,0)*(-1)).x( ) - mem(Space(0,-1,0)*(-1)).x( ) )
                        + beta_yz * ( mem(Space(0,0,1)*(-1)).x( ) - mem(Space(0,-1,1)*(-1)).x( ) )
                        + beta_yz * ( mem(Space(0,0,-1)*(-1)).x( ) - mem(Space(0,-1,-1)*(-1)).x( ) )
                        + beta_yx * ( mem(Space(1,0,0)*(-1)).x( ) - mem(Space(1,-1,0)*(-1)).x( ) )
                        + beta_yx * ( mem(Space(-1,0,0)*(-1)).x( ) - mem(Space(-1,-1,0)*(-1)).x( ) )
                        + delta_dir0 * ( mem(Space(0,1,0)*(-1)).x( ) - mem(Space(0,-2,0)*(-1)).x( ) )
                        ) * reci_dy;

                return float3_X( -curl_x, -curl_y, -curl_z );

                //return float3_X(diff(mem, 1).z() - diff(mem, 2).y(),
                //                diff(mem, 2).x() - diff(mem, 0).z(),
                //                diff(mem, 0).y() - diff(mem, 1).x());
            }
        };
    } // namespace leheSolver
} // namespace picongpu

