/* Copyright 2015-2019 Erik Zenker, Alexander Grund
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

#define BOOST_TEST_MODULE "PMacc Unit Tests"
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>


int main(int argc, char* argv[], char* envp[])
{
    int result = boost::unit_test::unit_test_main(&init_unit_test, argc, argv);

    return result;
}
