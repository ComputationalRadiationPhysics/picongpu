/* Copyright 2015-2019 Erik Zenker
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

// STL
#include <stdint.h> /* uint8_t */

// BOOST
#include <boost/test/unit_test.hpp>

// Boost.Test documentation: http://www.boost.org/doc/libs/1_59_0/libs/test/doc/html/index.html

/*******************************************************************************
 * Configuration
 ******************************************************************************/

// Nothing to configure, but here could be
// placed global variables, typedefs, classes.

/*******************************************************************************
 * Test Suite
 ******************************************************************************/
BOOST_AUTO_TEST_SUITE( template_unit_test )


/***************************************************************************
 * Test Cases
 ****************************************************************************/

// Normal test case
BOOST_AUTO_TEST_CASE( first ){
    BOOST_CHECK_EQUAL( sizeof(uint8_t), 1u );

}


BOOST_AUTO_TEST_SUITE_END()
