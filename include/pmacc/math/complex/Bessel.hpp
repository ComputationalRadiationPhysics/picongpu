/* Copyright 2003-2021 Alexander Debus, C. Bond
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file Bessel.hpp
 *
 *  Reference: Implementation is derived from a C++ implementation of
 *             complex Bessel functions from C. Bond (2003).
 *
 *  Original source downloaded from: http://www.crbond.com
 *  Download date: 2017/07/27
 *  Files: CBESSJY.CPP, BESSEL.H
 *  File-Header:
 *      cbessjy.cpp -- complex Bessel functions.
 *      Algorithms and coefficient values from "Computation of Special
 *      Functions", Zhang and Jin, John Wiley and Sons, 1996.
 *
 *     (C) 2003, C. Bond. All rights reserved.
 *
 *  The website (http://www.crbond.com) furthermore states:
 *  "This website contains a variety of materials related to
 *  technology and engineering. Downloadable software, much of it
 *  original, is available from some of the pages. All downloadable
 *  software is offered freely and without restriction -- although
 *  in most cases the files should be considered as works in progress
 *  (alpha or beta level). Source code is also included for some
 *  applications."
 *
 *  Code history:
 *  1/03 -- Added C/C++ source files for real and complex gamma function
 *  and psi function. Also added individual C/C++ files  for Bessel and
 *  modified Bessel functions of 1st and 2nd kinds for real and complex
 *  arguments. Updated Butterworth and Bessel filter tables with files of
 *   extended parameters including polynomials, poles and component values.
 *  6/04 -- Revised bessel.zip to correct errors in complex Bessel functions.
 *
 *  Further (re-)implementation of this code has been done by FZ Juelich.
 *  URL: http://apps.jcns.fz-juelich.de/redmine/issues/569#change-2056
 *  Above URL also includes a accuracy test report of this code against
 *  SLATEC, MAPLE and MATHEMATICA.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/algorithms/math.hpp"
#include "pmacc/algorithms/TypeCast.hpp"
#include "pmacc/math/Complex.hpp"


namespace pmacc
{
    namespace math
    {
        namespace bessel
        {
            template<typename T_Type, typename T_TableA, typename T_TableB, typename T_TableA1, typename T_TableB1>
            struct Cbesselj0Base;

            template<typename T_Type>
            HDINLINE typename J0<pmacc::math::Complex<T_Type>>::result j0(pmacc::math::Complex<T_Type> const& z)
            {
                return J0<pmacc::math::Complex<T_Type>>()(z);
            }

            template<typename T_Type, typename T_TableA, typename T_TableB, typename T_TableA1, typename T_TableB1>
            struct Cbesselj1Base;

            template<typename T_Type>
            HDINLINE typename J1<pmacc::math::Complex<T_Type>>::result j1(pmacc::math::Complex<T_Type> const& z)
            {
                return J1<pmacc::math::Complex<T_Type>>()(z);
            }

            PMACC_CONST_VECTOR(
                double,
                14,
                aDouble,
                -7.03125e-2,
                0.112152099609375,
                -0.5725014209747314,
                6.074042001273483,
                -1.100171402692467e2,
                3.038090510922384e3,
                -1.188384262567832e5,
                6.252951493434797e6,
                -4.259392165047669e8,
                3.646840080706556e10,
                -3.833534661393944e12,
                4.854014686852901e14,
                -7.286857349377656e16,
                1.279721941975975e19);

            PMACC_CONST_VECTOR(
                double,
                14,
                bDouble,
                7.32421875e-2,
                -0.2271080017089844,
                1.727727502584457,
                -2.438052969955606e1,
                5.513358961220206e2,
                -1.825775547429318e4,
                8.328593040162893e5,
                -5.006958953198893e7,
                3.836255180230433e9,
                -3.649010818849833e11,
                4.218971570284096e13,
                -5.827244631566907e15,
                9.476288099260110e17,
                -1.792162323051699e20);

            PMACC_CONST_VECTOR(
                double,
                14,
                a1Double,
                0.1171875,
                -0.1441955566406250,
                0.6765925884246826,
                -6.883914268109947,
                1.215978918765359e2,
                -3.302272294480852e3,
                1.276412726461746e5,
                -6.656367718817688e6,
                4.502786003050393e8,
                -3.833857520742790e10,
                4.011838599133198e12,
                -5.060568503314727e14,
                7.572616461117958e16,
                -1.326257285320556e19);

            PMACC_CONST_VECTOR(
                double,
                14,
                b1Double,
                -0.1025390625,
                0.2775764465332031,
                -1.993531733751297,
                2.724882731126854e1,
                -6.038440767050702e2,
                1.971837591223663e4,
                -8.902978767070678e5,
                5.310411010968522e7,
                -4.043620325107754e9,
                3.827011346598605e11,
                -4.406481417852278e13,
                6.065091351222699e15,
                -9.833883876590679e17,
                1.855045211579828e20);

            PMACC_CONST_VECTOR(
                float,
                14,
                aFloat,
                -7.03125e-2,
                0.112152099609375,
                -0.5725014209747314,
                6.074042001273483,
                -1.100171402692467e2,
                3.038090510922384e3,
                -1.188384262567832e5,
                6.252951493434797e6,
                -4.259392165047669e8,
                3.646840080706556e10,
                -3.833534661393944e12,
                4.854014686852901e14,
                -7.286857349377656e16,
                1.279721941975975e19);

            PMACC_CONST_VECTOR(
                float,
                14,
                bFloat,
                7.32421875e-2,
                -0.2271080017089844,
                1.727727502584457,
                -2.438052969955606e1,
                5.513358961220206e2,
                -1.825775547429318e4,
                8.328593040162893e5,
                -5.006958953198893e7,
                3.836255180230433e9,
                -3.649010818849833e11,
                4.218971570284096e13,
                -5.827244631566907e15,
                9.476288099260110e17,
                -1.792162323051699e20);

            PMACC_CONST_VECTOR(
                float,
                14,
                a1Float,
                0.1171875,
                -0.1441955566406250,
                0.6765925884246826,
                -6.883914268109947,
                1.215978918765359e2,
                -3.302272294480852e3,
                1.276412726461746e5,
                -6.656367718817688e6,
                4.502786003050393e8,
                -3.833857520742790e10,
                4.011838599133198e12,
                -5.060568503314727e14,
                7.572616461117958e16,
                -1.326257285320556e19);

            PMACC_CONST_VECTOR(
                float,
                14,
                b1Float,
                -0.1025390625,
                0.2775764465332031,
                -1.993531733751297,
                2.724882731126854e1,
                -6.038440767050702e2,
                1.971837591223663e4,
                -8.902978767070678e5,
                5.310411010968522e7,
                -4.043620325107754e9,
                3.827011346598605e11,
                -4.406481417852278e13,
                6.065091351222699e15,
                -9.833883876590679e17,
                1.855045211579828e20);
        } // namespace bessel
    } // namespace math
} // namespace pmacc
