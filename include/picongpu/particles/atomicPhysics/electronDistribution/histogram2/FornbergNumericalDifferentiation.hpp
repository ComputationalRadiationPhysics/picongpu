/* Copyright 2019-2020 Brian Marre
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

/** @file This file implements the generation of (finite difference)
 * (differentiation or interpolation) formula on arbitarily spaced grids,
 * as described by Fornberger in 1988.
 *
 * Intended usage as follow:
 * - specify all sample Points s_i relative to the evaluation point,
 * aka you specify s_i \in Q
 * with the actual sample points x_i being derived as follows:
 *
 * x_i = x + s_i * dx
 *
 * x  ... evaluation point
 * dx ... sample point scaling factor, arbitary
 *      remember s_i can be rational numbers, not only integers
 *
 * Sample Point Conditions/Assumptions:
 *  - the evaluation point must be included as s_0 = 0
 *  - all other sample points can be choosen freely
 *  - the order of sample points is arbitary but should not change once decided
 *      ,coefficents coorespond to sample points by the sample point index
 * - in principle s_i = (x_i - x) can also be used directly if scale invariance is
 *      not required,
 *      in principle it is even possible to use s_i = x_i, IF x = 0 is assured
 *
 * NOTE: sample points should be choosen with care to avoid the Runge phenomen
 *  choose chebyshev nodes if you can, to minimize runge's phenomen
 *
 * the coefficients c_i are used as follows:
 * f^(m)(x) = \sum_(nu=0)^(n) { c_nu * f( x + s_nu * dx ) * 1/dx } ;n >= m >= 0
 *
 * f ... function
 * m ... order of differentiation
 * n ... numSamplePoints
 *
 * Developer Notes:
 * ================
 * Notation:
 * ---------
 * For ease of usage the external interface uses longform names for variables,
 * desribing the intent of variables.
 *
 * The internal implementation does NOT follow this convention.
 * Instead it uses the derivation notation, sorry ;), conversion as follows:
 *
 *       m  ... T_orderDerivative
 *       n  ... numSamplePoints - 1, highest index of a sample point
 *       nu ... index
 *
 * All possible coefficents c is identified by their triple (m,n,nu).
 *  - If one or more indices are equal only the first one is listed as a different
 *    symbol, aka m==nu is always written as (m,n,m), NOT (nu,n,nu).
 *  - The least general description should almost always be used.
 *
 * Alogirthm Notes:
 * ----------------
 * (not a full description see out of source documentation for rest)
 *
 * The algorithm is based on 6 numbered recursion equations and the root c_(0,0,0) = 1.
 * <-> interpolation using only one sample point and s_0 = 0.
 *
 * The recursion equations, ordered by generality, are as follows:
 *  1: (m, n, nu) <- (m, n-1, nu)  "+" (m-1, n-1, nu)      ; 1 <= m <= n-1
 *  5: (m, n, n)  <- (m, n-1, n-1) "+" (m-1, n-1, n-1)     ; 1 <= m <= n-1
 *  3: (n, n, nu) <- (n-1, n-1, nu)                        ; n != nu
 *  2: (0, n, nu)<- (0, n-1, nu)                          ; n != nu
 *  6: (n, n, n)  <- (n-1, n-1, n-1)
 *  4: (0, n, n)  <- (0, n-1, n-1)
 *
 * - for both branching equations, 1 and 5, the first argumetn is labeled I,
 *   the second II.
 *
 * The recursion equations are implemented as private member functions,
 * called equation1 to equation6.
 * Their interfaces follow the same structure:
 *  T_Value equation# ( I , II , SamplePoints[], m, n, nu )
 *
 * - if argument is not present or superflous it is omitted.
 *
 * Coeffiencts are calculated ground up from the root using combinations of these
 * equation. The implementation is optimised for less memory used.
 *
 * The member function claculateWeights selects the best algorithm
 * The alogrithms are implemented as functions called getWeight
 *
 * The algorithm's use of recursions is documented in the form (m,n,nu)->(m',n',nu')
 * different arrows are used to differentiate between
 *      -> ... in place memory change
 *      => ... affecting other memory cell
 *
 * number marks are used to allow for quick navigation within the code
 */

#pragma once

#include <pmacc/algorithms/math.hpp>
#include <utility>


namespace picongpu
{
namespace particles
{
namespace atomicPhysics
{
namespace electronDistribution
{
namespace histogram2
{

    /** this class defines the numerical derivation of a function
     *
     * - provides weighting coefficents
     * - sample points stored by caller
     * - if storage of sample point also necessary create wrapper
     * - if different orders of derivative required get several instances
     *
     * @tparam T_numSamplePoints ... numer of sample points to be used, >= 1
     * @tparam T_orderDerivative ... order of derivative to be calculated, >= 0
     * @tparam T_Value ... datatype o coeffiecents,
     *           choose to match the return datatype of f
     * @tparam T_Argument ... datatype of argument space of x, should e float-like
     */
    template<
        typename T_Value,
        typename T_Argument
        >
    struct FornbergNumericalDifferentiation
    {

    public:
        // Notes on conversion:
        // m ... orderDerivative
        // n+1 ... T_numSamplePoints
        // nu ... index
        template<
            uint32_t T_numSamplePoints,
            uint32_t T_orderDerivative
        >
        static T_Value weighting(
            uint32_t const index,
            T_Argument const relativeSamplePoints[ T_numSamplePoints ]
            )
        {
            constexpr uint32_t m = T_orderDerivative;
            constexpr uint32_t n = T_numSamplePoints - 1u;
            uint32_t const nu = index;

            // check for bounds of input
            // 1.)+2.)+3.)
            if (
                ( m > n ) ||
                ( nu > n )
                )
                // not defined case
                return static_cast< T_Value >(0);

            // statements concerning m,n,nu so far
            // n >= m, nu >= 0

            // special cases:
            // 4.): case (m,m,nu)               ;else: m != n
            if ( m == n )
                //4.1.): (m, m, nu!=0)
                return getWeightMMNu< T_numSamplePoints >(
                    relativeSamplePoints,
                    m,
                    n,
                    nu
                    );

            // 5.): case (0,n,nu)               ;else: m != 0 and m != n
            if ( m == 0u )
                return getWeight0NNu< T_numSamplePoints >(
                    relativeSamplePoints,
                    m,
                    n,
                    nu
                    );

            // 6.): case (m,n,n)                ;else: n!= nu and m != 0 and m != n
            if ( n == nu )
            {
                // choose construction method with lowest memory
                if ( m > ( n - m ) )
                    // 6.1.): case: m > n-m
                    return getWeightMNN_1< T_numSamplePoints >(
                        relativeSamplePoints,
                        m,
                        n,
                        nu
                        );
                else
                    // 6.2.): case m < n-m
                    return getWeightMNN_2< T_numSamplePoints >(
                        relativeSamplePoints,
                        m,
                        n,
                        nu
                        );
            }

            // general cases

            // statements concerning m,n,nu
            // n > m > 0 and n > nu >= 0
            // unknown wether m > nu

            // 7.): case (m,n,nu)
            if ( m >= nu )
            {
                // 7.1.) case (m,n,nu): n > m >= nu >= 0
                // => (m,n,nu) = (nu+o, nu+o+k, nu)


                // 7.1.1.) k > nu
                if ( n - m > nu )
                {
                    // choose construction method with lowest memory
                    if ( n - m > m )
                    {
                        // 7.1.1.1.)
                        return getWeightMNNu_o_ko_0_gt_1< T_numSamplePoints >(
                            relativeSamplePoints,
                            m,
                            n,
                            nu
                            );
                    }
                    else
                    {
                        // 7.1.1.2.)
                        return getWeightMNNu_o_ko_0_gt_2< T_numSamplePoints >(
                            relativeSamplePoints,
                            m,
                            n,
                            nu
                            );
                    }
                }
                else
                {
                    // 7.1.2.) k <= nu; m >= nu
                    if ( n - m > m )
                    {
                        // 7.1.2.1.)
                        return getWeightMNNu_o_ko_0_le_1< T_numSamplePoints >(
                            relativeSamplePoints,
                            m,
                            n,
                            nu
                            );
                    }
                    else
                    {
                        // 7.1.2.2.)
                        return getWeightMNNu_o_ko_0_le_2< T_numSamplePoints >(
                            relativeSamplePoints,
                            m,
                            n,
                            nu
                            );
                    }
                }
            }

            // => m < nu
            // 8.): case (m,n,nu) ;n > nu > m > 0
            // => (m,n,nu) = (m, m+o+k, m+o)

            // 8.1.)
            if ( n - nu <= m )
            {
                // 8.1.1)
                // choose construction method with lowest memory required
                if ( ( n - m ) < m )
                {
                    // 8.1.1.1.)
                    return getWeightMNNu_0_ok_o_le_1< T_numSamplePoints >(
                        relativeSamplePoints,
                        m,
                        n,
                        nu
                        );
                }
                else
                {
                    // 8.1.1.2.)
                    return getWeightMNNu_0_ok_o_le_2< T_numSamplePoints >(
                        relativeSamplePoints,
                        m,
                        n,
                        nu
                        );
                }

            }
            // 8.2.) ;k > m
            else
            {
                //8.2.1.)
                // choose construction method with lowest memory required
                if ( n - m < m )
                {
                    // 8.2.1.1.)
                    return getWeightMNNu_0_ko_o_gt_1< T_numSamplePoints >(
                        relativeSamplePoints,
                        m,
                        n,
                        nu
                        );
                }
                else
                {
                    // 8.2.1.2.)
                    return getWeightMNNu_0_ko_o_gt_2< T_numSamplePoints >(
                        relativeSamplePoints,
                        m,
                        n,
                        nu
                        );
                }
            }
        }

    private:

        template<
            uint32_t T_numSamplePoints
        >
        static T_Value equation1( 
            T_Value const I,
            T_Value const II,
            T_Argument const alpha[ T_numSamplePoints ],
            uint32_t const n,
            uint32_t const m,
            uint32_t const nu
            )
        {
            return static_cast< T_Value >(
                1._X/static_cast< T_Value >( alpha[ n ] - alpha[ nu ] )
                * ( I * alpha[ n ] - m * II )
                );
        }
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value equation2( 
            T_Value const I,
            T_Argument const alpha[ T_numSamplePoints ],
            uint32_t const n,
            uint32_t const nu
            )
        {
            return static_cast< T_Value >(
                alpha[ n ]/
                    ( alpha[ n ] - alpha[ nu ] )
                    * I
                );
        }

        template<
            uint32_t T_numSamplePoints
        >
        static T_Value equation3( 
            T_Value const I,
            T_Argument const alpha[ T_numSamplePoints ],
            uint32_t const n,
            uint32_t const nu
            )
        {
            return static_cast< T_Value >(
                (n) /
                    ( alpha[ nu ] - alpha[ n ] )
                * I
                );
        }

        /** w_n(x) = prod_(i=0)^(n) (x - alpha_i)
         */
        template<
            uint32_t T_numSamplePoints
        >
        static T_Argument w(
            T_Argument const x,
            uint32_t const n,
            T_Argument const relativeSamplePoints[ T_numSamplePoints ]
            )
        {
            T_Argument result = static_cast< T_Argument >( 1. );

            if ( n > T_numSamplePoints )
            {
                return static_cast< T_Argument >( 0 );
            }

            for ( uint32_t i = 0u; i <= n; i++ )
            {
                result *= ( x - relativeSamplePoints[ i ] );
            }

            return result;
        }

        template<
            uint32_t T_numSamplePoints
        >
        static T_Value equation4(
            T_Value const I,
            T_Argument const alpha[ T_numSamplePoints ],
            uint32_t const n
            )
        {
            return static_cast< T_Value >(
                w(
                    alpha[ n - 1u ],
                    n - 2u,
                    alpha
                    )
                / w(
                    alpha[ n ],
                    n - 1u,
                    alpha
                    )
                * alpha[ n - 1u ] * I
                );
        }

        template<
            uint32_t T_numSamplePoints
        >
        static T_Value equation5( 
            T_Value const I,
            T_Value const II,
            T_Argument const alpha[ T_numSamplePoints ],
            uint32_t const n,
            uint32_t const m
            )
        {
            return static_cast< T_Value >(
                w(
                    alpha[ n - 1u ],
                    n - 2u,
                    alpha
                    )
                /w(
                    alpha[ n ],
                    n - 1u,
                    alpha
                    )
                * ( m * II - alpha[ n - 1u ] * I )
                );
        }

        template<
            uint32_t T_numSamplePoints
        >
        static T_Value equation6( 
            T_Value const I,
            T_Argument const alpha[ T_numSamplePoints ],
            uint32_t const m
            )
        {
            return static_cast< T_Value >(
                I * m
                );
        }

        // m ... T_orderDerivative
        // n+1 ... T_numSamplePoints
        // nu ... index

        //4.): case (m,m,nu)
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMMNu (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            // init with (0,0,0)
            T_Value weight = static_cast< T_Value >(1);

            for ( uint32_t i = 1u; i <= nu; i++ )
            {
                // (i-1,i-1,i-1) -> (i,i,i)
                weight = equation6< T_numSamplePoints >(
                    weight,
                    relativeSamplePoints,
                    i
                    );
            }
            for ( uint32_t i = nu + 1u; i <= m; i++)
                {
                    // (i-1,i-1,nu) -> (i,i,nu)
                    weight = equation3< T_numSamplePoints >(
                        weight,
                        relativeSamplePoints,
                        i,
                        nu
                        );
                }
                return weight;
        }

        //5.): case (0,n,nu)
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeight0NNu (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            // init with (0,0,0)
            T_Value weight = static_cast< T_Value >(1);

            for ( uint32_t i = 1u; i <= nu; i++ )
            {
                // (0,i-1,i-1) -> (0,i,i)
                weight = equation4< T_numSamplePoints >(
                    weight,
                    relativeSamplePoints,
                    i);
            }
            for ( uint32_t j = nu + 1u; j <= n; j++ )
            {
                // (0, j-1, nu) -> (0, j, nu)
                weight = equation2< T_numSamplePoints >(
                    weight,
                    relativeSamplePoints,
                    j,
                    nu
                    );
            }
            return weight;
        }

        // 6.1.): case (m,n,n) ;m > n-m
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNN_1 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - m;

            T_Value weights[ k + 1u ] = {};

            // init root with (0,0,0)
            weights[0] = static_cast< T_Value >(1);

            // init base collumn with (0,i,i)
            for ( uint32_t i = 1u; i <= k; i++ )
            {
                // (0,i-1,i-1) => (0,i,i)
                weights[ i ] = equation4< T_numSamplePoints >(
                    weights[ i - 1 ],
                    relativeSamplePoints,
                    i
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1; j <= m; j++ )
            {
                // (j-1,j-1,j-1) -> (j,j,j)
                weights[0] = equation6< T_numSamplePoints >(
                    weights[0],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= k; i++)
                {
                    // weights_i     = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( j  , (i-1) + j, (i-1) + j )
                    // ->weights_i   = ( j, i + j, i + j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1 ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i
                        );
                }
            }
            return weights[k];
        }

        // 6.2.): case (m,n,n) ;m < n-m
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNN_2 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - m;

            T_Value weights[ m + 1u ] = {};

            // init root with (0,0,0)
            weights[0] = static_cast< T_Value >(1);

            // init base collumn with (i,i,i)
            for ( uint32_t i = 1u; i <= m; i++ )
            {
                // (i-1,i-1,i-1) => (i,i,i)
                weights[ i ] = equation6< T_numSamplePoints >(
                    weights[ i - 1 ],
                    relativeSamplePoints,
                    i);
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1; j <= k; j++ )
            {
                // (0,j-1,j-1) -> (0,j,j)
                weights[0] = equation4< T_numSamplePoints >(
                    weights[0],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= m; i++)
                {
                    // weights_i     = ( i  , i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( i-1, (i-1) + j, (i-1) + j )
                    // ->weights_i   = ( i  , i + j, i + j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1 ],
                        relativeSamplePoints,
                        j,
                        i
                        );
                }
            }
            return weights[m];
        }

        // 7.1.1.1): case (m,n,nu) ;m > nu; k > nu; k > o+nu+1
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_o_ko_0_gt_1 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - m;
            uint32_t const o = m - nu;

            T_Value weights[ o + nu + 1u ]  = {};

            // init root with (0,0,0)
            weights[0] = static_cast< T_Value >(1);

            // init base collumn with (i,i,i) until (nu,nu,nu)
            for ( uint32_t i = 1u; i <= nu ; i++ )
            {
                // (i-1,i-1,i-1) => (i,i,i)
                weights[ i ] = equation6< T_numSamplePoints >(
                    weights[ i - 1 ],
                    relativeSamplePoints,
                    i
                    );
            }
            // init base collumn with (i,i,nu) until (m,m,nu)
            for ( uint32_t i = nu + 1; i <= o + nu; i++ )
            {
                // ( i-1, i-1, nu ) => ( i, i, nu )
                weights[ i ] = equation3< T_numSamplePoints >(
                    weights[ i - 1 ],
                    relativeSamplePoints,
                    i,
                    nu
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= nu ; j++ )
            {
                // starting line
                // (0,j-1,j-1) -> (0,j,j)
                weights[ 0 ] = equation4< T_numSamplePoints >(
                    weights[0],
                    relativeSamplePoints,
                    j
                    );

                // first nu-1 lines
                for ( uint32_t i = 1u; i <= (nu - 1u); i++ )
                {
                    if ( i + j <= nu)
                    {
                        // weights_i      = ( i  , i + (j-1), i + (j-1) )
                        // weights_(i-1)  = ( i-1, (i-1) + j, (i-1) + j )
                        // ->weights_i    = ( i  , i + j, i + j )
                        weights[ i ] = equation5< T_numSamplePoints >(
                            weights[ i ],
                            weights[ i - 1u ],
                            relativeSamplePoints,
                            i,
                            i + j
                            );
                    }
                    else
                    {
                        // weights_i     = ( i  , i + (j-1), nu )
                        // weights_(i-1) = ( i-1, (i-1) + j, nu )
                        // -> weights_i  = ( i  , i + j, nu )
                        weights[ i ] = equation1< T_numSamplePoints >(
                            weights[ i ],
                            weights[ i - 1u ],
                            relativeSamplePoints,
                            i,
                            i + j,
                            nu
                            );
                    }
                }
                // l+1 rest lines
                for ( uint32_t i = nu ; i <= o + nu; i++ )
                {
                    // weights_i      = ( i  , i + (j-1), nu )
                    // weights_(i-1)  = ( i-1, (i-1) + j, nu )
                    // ->weights_i    = ( i  , i + j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights [ i ],
                        weights [ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i + j,
                        nu
                        );
                }
            }

            // remaining steps
            for ( uint32_t j = nu + 1u; j <= k; j++ )
            {
                // (0, j-1, nu) -> (0,j,nu)
                weights[ 0 ] = equation2( weights[ 0 ], relativeSamplePoints, j, nu );
                for ( uint32_t i = 1u; i <= nu + o; i++ )
                {
                    // weights_i      = ( i  , i + (j-1), nu )
                    // weights_(i-1)  = ( i-1, (i-1) + j, nu )
                    // ->weights_i    = ( i  , i + j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i + j,
                        nu
                        );
                }
            }
            return weights[o + nu];
        }

        // 7.1.1.2): case (m,n,nu) ;m > nu; k > nu; k < o+nu+1
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_o_ko_0_gt_2 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - m;
            uint32_t const o = m - nu;

            T_Value weights[ k + 1u ]  = {};

            // init root with (0,0,0)
            weights[0] = static_cast< T_Value >(1);

            // init base collumn with (0,i,i) until (0,nu,nu)
            for ( uint32_t i = 1u; i <= nu ; i++ )
            {
                // (0,i-1,i-1) => (0,i,i)
                weights[ i ] = equation4< T_numSamplePoints >(
                    weights[ i - 1u ],
                    relativeSamplePoints,
                    i
                    );
            }
            // init base collumn with (0,i,nu) until (0,m,nu)
            for ( uint32_t i = nu + 1u; i <= o; i++ )
            {
                // ( 0, i-1, nu ) => ( 0, i, nu )
                weights[ i ] = equation2< T_numSamplePoints >(
                    weights[ i - 1u ],
                    relativeSamplePoints,
                    i,
                    nu
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= nu ; j++ )
            {
                // starting line
                // (j-i,j-1,j-1) -> (j,j,j)
                weights[ 0 ] = equation6< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= (nu - j); i++ )
                {
                    // weights_i      = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1)  = ( j  , (i-1) + j, (i-1) + j )
                    // ->weights_i    = ( j  , i + j, i + j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1 ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j
                        );
                }
                for ( uint32_t i = nu - j + 1u; i <= k; i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), nu )
                    // weights_(i-1) = ( j  , (i-1) + j, nu )
                    // -> weights_i  = ( j  , i + j, nu )
                        weights[ i ] = equation1< T_numSamplePoints >(
                            weights[ i - 1 ],
                            weights[ i ],
                            relativeSamplePoints,
                            j,
                            i + j,
                            nu
                            );
                }
            }
            for ( uint32_t j = nu + 1u ; j <= o + nu; j++)
            {
                // starting line
                // (j-i,j-1,nu) -> (j,j,nu)
                weights[ 0 ] = equation3< T_numSamplePoints >(
                    weights [ 0 ],
                    relativeSamplePoints,
                    j,
                    nu
                    );

                for ( uint32_t i = 1u; i <= k; i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), nu )
                    // weights_(i-1) = ( j  , (i-1) + j, nu )
                    // -> weights_i  = ( j  , i + j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i - 1 ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j,
                        nu
                        );
                }
            }
            return weights[k];
        }

        // 7.1.2.1): case (m,n,nu) ;m > nu; k <= nu; k > o + nu + 1
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_o_ko_0_le_1 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - m;
            uint32_t const o = m - nu;

            T_Value weights[ o + nu + 1u ]  = {};

            // init root with (0,0,0)
            weights[0] = static_cast< T_Value >(1);

            // init base collumn with (i,i,i) until (nu,nu,nu)
            for ( uint32_t i = 1u; i <= nu ; i++ )
            {
                // (i-1,i-1,i-1) => (i,i,i)
                weights[ i ] = equation6< T_numSamplePoints >(
                    weights[ i - 1 ],
                    relativeSamplePoints,
                    i
                    );
            }
            // init base collumn with (i,i,nu) until (m,m,nu)
            for ( uint32_t i = nu + 1; i <= o + nu; i++ )
            {
                // ( i-1, i-1, nu ) => ( i, i, nu )
                weights[ i ] = equation3< T_numSamplePoints >(
                    weights[ i - 1 ],
                    relativeSamplePoints,
                    i,
                    nu
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= k ; j++ )
            {
                // starting line
                // (0,j-1,j-1) -> (0,j,j)
                weights[ 0 ] = equation4< T_numSamplePoints >(
                    weights[0],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= (nu - k); i++ )
                {
                    // weights_i      = ( i  , i + (j-1), i + (j-1) )
                    // weights_(i-1)  = ( i-1, (i-1) + j, (i-1) + j )
                    // ->weights_i    = ( i  , i + j, i + j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i + j
                        );
                }
                for ( uint32_t i = (nu - k) + 1u; i <= nu; i++ )
                {
                    if ( i + j <= nu )
                    {
                        // weights_i      = ( i  , i + (j-1), i + (j-1) )
                        // weights_(i-1)  = ( i-1, (i-1) + j, (i-1) + j )
                        // ->weights_i    = ( i  , i + j, i + j )
                        weights[ i ] = equation5< T_numSamplePoints >(
                            weights[ i ],
                            weights[ i - 1u ],
                            relativeSamplePoints,
                            i,
                            i + j
                            );
                    }
                    else
                    {
                        // weights_i     = ( i  , i + (j-1), nu )
                        // weights_(i-1) = ( i-1, (i-1) + j, nu )
                        // -> weights_i  = ( i  , i + j, nu )
                        weights[ i ] = equation1< T_numSamplePoints >(
                            weights[ i ],
                            weights[ i - 1u ],
                            relativeSamplePoints,
                            i,
                            i + j,
                            nu
                            );
                    }
                }
                for ( uint32_t i = nu + 1u; i <= nu + o; i++ )
                {
                    // weights_i      = ( i  , i + (j-1), nu )
                    // weights_(i-1)  = ( i-1, (i-1) + j, nu )
                    // ->weights_i    = ( i  , i + j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights [ i ],
                        weights [ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i + j,
                        nu
                        );
                }
            }
            return weights[ o + nu ];
        }

        // 7.1.2.2): case (m,n,nu) ;m > nu; k <= nu; k < o+nu+1
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_o_ko_0_le_2 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - m;
            uint32_t const o = m - nu;

            T_Value weights[ k + 1u ] = {};

            // init root with (0,0,0)
            weights[0] = static_cast< T_Value >(1);

            // init base collumn with (0,i,i) until (0,nu,nu)
            for ( uint32_t i = 1u; i <= k ; i++ )
            {
                // (0,i-1,i-1) => (0,i,i)
                weights[ i ] = equation4< T_numSamplePoints >(
                    weights[ i - 1u ],
                    relativeSamplePoints,
                    i
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= nu - k; j++ )
            {
                // starting line
                // (j-i,j-1,j-1) -> (j,j,j)
                weights[ 0 ] = equation6< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= k; i++ )
                {
                    // weights_i      = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1)  = ( j  , (i-1) + j, (i-1) + j )
                    // ->weights_i    = ( j  , i + j, i + j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j
                        );
                }
            }
            for ( uint32_t j = nu - k + 1u; j <= nu; j++ )
            {
                // starting line
                // (j-i,j-1,j-1) -> (j,j,j)
                weights[ 0 ] = equation6(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= nu - j; i++ )
                {
                    // weights_i      = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1)  = ( j  , (i-1) + j, (i-1) + j )
                    // ->weights_i    = ( j  , i + j, i + j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j
                        );
                }
                for ( uint32_t i = nu - j + 1u; i <= k; i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), nu )
                    // weights_(i-1) = ( j  , (i-1) + j, nu )
                    // -> weights_i  = ( j  , i + j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i+j,
                        nu
                        );
                }
            }

            for ( uint32_t j = nu + 1u ; j <= o + nu; j++)
            {
                for ( uint32_t i = 1u; i <= k; i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), nu )
                    // weights_(i-1) = ( j  , (i-1) + j, nu )
                    // -> weights_i  = ( j  , i+j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i+j,
                        nu
                        );
                }
            }
            return weights[k];
        }

        // 8.1.1.1.): case (m,n,nu) ;m < nu; k <= m; k+o < m
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_0_ok_o_le_1 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - nu;
            uint32_t const o = nu - m;

            T_Value weights[ o + k + 1u ]  = {};

            // init root with (0,0,0)
            weights[ 0 ] = static_cast< T_Value >( 1u );

            // init base collumn
            for ( uint32_t i = 1u; i <= k + o; i++ )
            {
                // (0,i-1,i-1) => (0,i,i)
                weights[ i ] = equation4< T_numSamplePoints >(
                    weights[ i - 1u ],
                    relativeSamplePoints,
                    i
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= m - k; j++ )
            {
                // starting line
                // (j-1, j-1, j-1) -> (j,j,j)
                weights[ 0 ] = equation6< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= k + o; i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( j  , (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( j  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j
                        );
                }
            }

            for ( uint32_t j = m - k + 1u; j <= m; j++ )
            {
                // starting line 
                // (j-1,j-1,j-1) -> (j,j,j)
                weights[ 0 ] = equation6< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= ( m - j + o ); i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( j  , (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( j  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j
                        );
                }
                for ( uint32_t i = m - j + o + 1u; i <= k + o; j++ )
                {
                    // weights_i     = ( j-1, i + (j-1), nu )
                    // weights_(i-1) = ( j  , (i-1) + j, nu )
                    // -> weights_i  = ( j  , i+j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j,
                        nu
                        );
                }
            }
            return weights[ o + k ];
        }

        // 8.1.1.2.): case (m,n,nu) ;m < nu; k <= m; k+o >= m
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_0_ok_o_le_2 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - nu;
            uint32_t const o = nu - m;

            T_Value weights[ m + 1u ] = {};

            // init root with (0,0,0)
            weights[ 0 ] = static_cast< T_Value >( 1u );

            // init base collumn
            for ( uint32_t i = 1u; i <= m; i++ )
            {
                // (i-1,i-1,i-1) => (i,i,i)
                weights[ i ] = equation6< T_numSamplePoints >(
                    weights[ i -1 ],
                    relativeSamplePoints,
                    i
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= o; j++ )
            {
                // starting line
                // (0, j-1, j-1) -> (0,j,j)
                weights[ 0 ] = equation4< T_numSamplePoints >(
                    weights [ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= m; i++ )
                {
                    // weights_i     = ( i  , i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( i-1, (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( i  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i+j
                        );
                }
            }

            for ( uint32_t j = o + 1u; j <= k + o; j++ )
            {
                // starting line
                // (0, j-1, j-1) -> (0,j,j)
                weights[ 0 ] = equation4< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= m - j + o; j++ )
                {
                    // weights_i     = ( i  , i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( i-1, (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( i  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1 ],
                        relativeSamplePoints,
                        i,
                        i + j
                        );
                }
                for ( uint32_t i = m - j + o + 1u; i <= m; i++ )
                {
                    // weights_i     = ( i  , i + (j-1), nu )
                    // weights_(i-1) = ( i-1, (i-1) + j, nu )
                    // -> weights_i  = ( i  , i+j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights [ i ],
                        weights[ i - 1],
                        relativeSamplePoints,
                        i,
                        i + j,
                        nu
                        );
                }
            }
            return weights[ m ];
        }

        // 8.1.2.1.): case (m,n,nu) ;m < nu; k > m; k+o < m
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_0_ko_o_gt_1 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - nu;
            uint32_t const o = nu - m;

            T_Value weights[ o + k + 1u];

            // init root with (0,0,0)
            weights[ 0 ] = static_cast< T_Value >( 1u );

            // init base collumn
            for ( uint32_t i = 1u; i <= m + o; i++ )
            {
                // (0,i-1,i-1) => (0,i,i)
                weights[ i ] = equation4< T_numSamplePoints >(
                    weights[ i - 1u ],
                    relativeSamplePoints,
                    i
                    );
            }
            for ( uint32_t i = m + o + 1u; i <= k + o; i++ )
            {
                // (0,i-1,nu) => (0,i,nu)
                weights[ i ] = equation2< T_numSamplePoints >(
                    weights[ i - 1u ],
                    relativeSamplePoints,
                    i,
                    nu
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= m; j++ )
            {
                // starting line
                // (j-1, j-1, j-1) -> (j,j,j)
                weights[ 0 ] = equation6< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= o + m - j; i++ )
                {
                    // weights_i     = ( j-1, i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( j  , (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( j  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j
                        );
                }
                for ( uint32_t i = m - j + o + 1u; i <= k + o; j++ )
                {
                    // weights_i     = ( j-1, i + (j-1), nu )
                    // weights_(i-1) = ( j  , (i-1) + j, nu )
                    // -> weights_i  = ( j  , i+j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i - 1u ],
                        weights[ i ],
                        relativeSamplePoints,
                        j,
                        i + j,
                        nu
                        );
                }
            }
            return weights[ k + o ];
        }

        // 8.1.2.2.): case (m,n,nu) ;m < nu; k > m; k+o > m
        template<
            uint32_t T_numSamplePoints
        >
        static T_Value getWeightMNNu_0_ko_o_gt_2 (
            T_Argument const relativeSamplePoints[ T_numSamplePoints ],
            uint32_t const m,
            uint32_t const n,
            uint32_t const nu
        )
        {
            uint32_t const k = n - nu;
            uint32_t const o = nu - m;

            T_Value weights[ m + 1u ] = {};

            // init root with (0,0,0)
            weights[ 0 ] = static_cast< T_Value >( 1u );

            // init base collumn
            for ( uint32_t i = 1u; i <= m; i++ )
            {
                // (i-1,i-1,i-1) => (i,i,i)
                weights[ i ] = equation6< T_numSamplePoints >(
                    weights[ i -1 ],
                    relativeSamplePoints,
                    i
                    );
            }

            // advance collumn piecewise, in place
            for ( uint32_t j = 1u; j <= o; j++ )
            {
                // starting line
                // (0, j-1, j-1) -> (0,j,j)
                weights[ 0 ] = equation4< T_numSamplePoints >(
                    weights [ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= m; i++ )
                {
                    // weights_i     = ( i  , i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( i-1, (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( i  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i+j
                        );
                }
            }

            for ( uint32_t j = o + 1u; j <= m + o; j++ )
            {
                // starting line
                // (0, j-1, j-1) -> (0,j,j)
                weights[ 0 ] = equation4< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j
                    );

                for ( uint32_t i = 1u; i <= m - j + o; j++ )
                {
                    // weights_i     = ( i  , i + (j-1), i + (j-1) )
                    // weights_(i-1) = ( i-1, (i-1) + j, (i-1) + j )
                    // -> weights_i  = ( i  , i+j, i+j )
                    weights[ i ] = equation5< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1 ],
                        relativeSamplePoints,
                        i,
                        i + j
                        );
                }

                for ( uint32_t i = m - j + o + 1u; i <= m; i++ )
                {
                    // weights_i     = ( i  , i + (j-1), nu )
                    // weights_(i-1) = ( i-1, (i-1) + j, nu )
                    // -> weights_i  = ( i  , i+j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights [ i ],
                        weights[ i - 1],
                        relativeSamplePoints,
                        i,
                        i + j,
                        nu
                        );
                }
            }

            for ( uint32_t j = m + o + 1u; j <= k + o; j++ )
            {
                // starting line
                // (0,j-1, nu) -> (0,j,nu)
                weights[ 0 ] = equation2< T_numSamplePoints >(
                    weights[ 0 ],
                    relativeSamplePoints,
                    j,
                    nu
                    );

                for ( uint32_t i = 1u; i <= m; i++ )
                {
                    // weights_i     = ( i  , i + (j-1), nu )
                    // weights_(i-1) = ( i-1, (i-1) + j, nu )
                    // -> weights_i  = ( i  , i+j, nu )
                    weights[ i ] = equation1< T_numSamplePoints >(
                        weights[ i ],
                        weights[ i - 1u ],
                        relativeSamplePoints,
                        i,
                        i + j,
                        nu
                        );
                }
            }
            return weights[ m ];
        }
    };

} // namespace histogram2
} // namespace electronDistribution
} // namespace atomicPhysics
} // namespace particles
} // namespace picongpu
