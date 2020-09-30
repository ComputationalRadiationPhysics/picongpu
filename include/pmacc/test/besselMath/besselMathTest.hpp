/* Copyright 2016-2017 Alexander Grund, Alexander Debus
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"

#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <set>
#include <algorithm>
#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/Complex.hpp"
#include <stdint.h>

BOOST_AUTO_TEST_SUITE( besselMath )

namespace bmpl = boost::mpl;

PMACC_CONST_VECTOR( float, 14, bFloat,
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
    -1.792162323051699e20
);

PMACC_CONST_VECTOR(float, 14, a1Float,
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
    -1.326257285320556e19
);

PMACC_CONST_VECTOR( float, 14, b1Float,
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
     1.855045211579828e20
);

namespace
{
    template<
        uint32_t T_numWorkers,
        uint32_t T_numCalcsPerBlock
    >
    struct CalculateBessel
    {
        template< class T_Box, typename T_Acc >
        HDINLINE void operator()( const T_Acc & acc, T_Box outputbox, uint32_t numThreads ) const
        {
            namespace math = ::pmacc::algorithms::math;
            using complex_64 = ::pmacc::math::Complex< double >;

            using namespace pmacc::mappings::threads;
            constexpr uint32_t numWorkers = T_numWorkers;

            uint32_t const workerIdx = threadIdx.x;

            uint32_t const blockId = blockIdx.x * T_numCalcsPerBlock;
            ForEachIdx<
                IdxConfig<
                    T_numCalcsPerBlock,
                    numWorkers
                >
            >{ workerIdx }(
                [&](
                    uint32_t const linearId,
                    uint32_t const
                )
                {
                    uint32_t const localId = blockId + linearId;
                    if( localId < numThreads )
                    {
                        outputbox( localId ) = ( math::bessel::j0( complex_64( double( 1.2 ), double( 3.4 ) ) ) );
                    }
                }
            );


        }
    };
}

struct BesselMathTest
{
    void operator()()
    {
        double x = 1.2;
        double eps = 1.0e-13;

        namespace math = ::pmacc::algorithms::math;
        using complex_64 = ::pmacc::math::Complex< double >;

        /* Test real-valued Bessel-functions. This just tests calling math-libraries. */
        double res = math::bessel::j0( x );
        BOOST_REQUIRE_CLOSE( res, double( 0.6711327442643626 ), eps );
        res = math::bessel::j1( x );
        BOOST_REQUIRE_CLOSE( res, double( 0.4982890575672156 ), eps );
        res = math::bessel::i0( x );
        BOOST_REQUIRE_CLOSE( res, double( 1.393725584134064 ), eps );
        res = math::bessel::i1( x );
        BOOST_REQUIRE_CLOSE( res, double( 0.7146779415526429 ), eps );
        res = math::bessel::y0( x );
        BOOST_REQUIRE_CLOSE( res, double( 0.22808350322719695 ), eps );
        res = math::bessel::y1( x );
        BOOST_REQUIRE_CLOSE( res, double( -0.621136379748848 ), eps );
        res = math::bessel::jn( 5, x );
        BOOST_REQUIRE_CLOSE( res, double( 0.0006101049237489683 ), eps );
        res = math::bessel::yn( 5, x );
        BOOST_REQUIRE_CLOSE( res, double( -107.65134933876895 ), eps );

        /* Tests custom implementation of complex-valued Bessel functions. Here one tests all the branches of the code that are responsible for different modulus-values of the initial complex number. */
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 1.2 ), double( 3.4 ) ) ) ).get_real( ) , double( 3.460244764532168 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 1.2 ), double( 3.4 ) ) ) ).get_imag( ) , double( -5.544948494375138 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 1.2 ), double( 3.4 ) ) ) ).get_real( ) , double( 4.966740719480319 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 1.2 ), double( 3.4 ) ) ) ).get_imag( ) , double( 2.654040063652766 ), eps );

        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( -1.2 ), double( 3.4 ) ) ) ).get_real( ) , double( 3.460244764532168 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( -1.2 ), double( 3.4 ) ) ) ).get_imag( ) , double( 5.544948494375138 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( -1.2 ), double( 3.4 ) ) ) ).get_real( ) , double( -4.966740719480319 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( -1.2 ), double( 3.4 ) ) ) ).get_imag( ) , double( 2.654040063652766 ), eps );

        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 12. ), double( 34. ) ) ) ).get_real( ) , double( 2.879462096290229e+13 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 12. ), double( 34. ) ) ) ).get_imag( ) , double( 2.614471411315435e+13 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 12. ), double( 34. ) ) ) ).get_real( ) , double( -2.5666122527960234e+13 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 12. ), double( 34. ) ) ) ).get_imag( ) , double( 2.8538187928343023e+13 ), eps );

        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( -12. ), double( 34. ) ) ) ).get_real( ) , double( 2.879462096290229e+13 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( -12. ), double( 34. ) ) ) ).get_imag( ) , double( -2.614471411315435e+13 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( -12. ), double( 34. ) ) ) ).get_real( ) , double( 2.5666122527960234e+13 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( -12. ), double( 34. ) ) ) ).get_imag( ) , double( 2.8538187928343023e+13 ), eps );

        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 12. ), double( 17. ) ) ) ).get_real( ) , double( 1.3570554101490807e+6 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 12. ), double( 17. ) ) ) ).get_imag( ) , double( 1.632616877799723e+6 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 12. ), double( 17. ) ) ) ).get_real( ) , double( -1.5812249847121988e+6 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 12. ), double( 17. ) ) ) ).get_imag( ) , double( 1.3533807423250647e+6 ), eps );

        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 102. ), double( 304. ) ) ) ).get_real( ) , double( 6.161900976274428e+129 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j0( complex_64( double( 102. ), double( 304. ) ) ) ).get_imag( ) , double( -2.2818659357268304e+130 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 102. ), double( 304. ) ) ) ).get_real( ) , double( 2.2787965009502116e+130 ), eps );
        BOOST_REQUIRE_CLOSE( ( math::bessel::j1( complex_64( double( 102. ), double( 304. ) ) ) ).get_imag( ) , double( 6.141450635196441e+129 ), eps );

        constexpr uint32_t numBlocks = 1;
        constexpr uint32_t numCalcsPerBlock = 32;
        constexpr uint32_t numThreads = numBlocks * numCalcsPerBlock;

        pmacc::HostDeviceBuffer<complex_64, 1> buffer(numThreads);
        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            numCalcsPerBlock
        >::value;

        PMACC_KERNEL( CalculateBessel<
            numWorkers,
            numCalcsPerBlock
        >{  })(
            numBlocks,
            numWorkers
        )(
            buffer.getDeviceBuffer().getDataBox(),
            numThreads
        );
        buffer.deviceToHost();
        auto hostBox = buffer.getHostBuffer().getDataBox();
        // Make sure they are the same
        for(uint32_t i=0; i<numThreads; i++)
        {
            BOOST_REQUIRE_CLOSE(
                  ( math::bessel::j0( complex_64( double( 1.2 ), double( 3.4 ) ) ) ).get_real( ),
                  hostBox(i).get_real( ),
                  eps 
                );
            BOOST_REQUIRE_CLOSE(
                  ( math::bessel::j0( complex_64( double( 1.2 ), double( 3.4 ) ) ) ).get_imag( ),
                  hostBox(i).get_imag( ),
                  eps 
                );
        }
     }
};

BOOST_AUTO_TEST_CASE( testBesselMath )
{
    BesselMathTest()();
}

BOOST_AUTO_TEST_SUITE_END()

