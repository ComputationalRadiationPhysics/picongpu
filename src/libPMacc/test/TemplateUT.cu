// STL
#include <stdint.h> /* uint8_t */
#include <stdio.h>  /* printf */

// BOOST
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

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
