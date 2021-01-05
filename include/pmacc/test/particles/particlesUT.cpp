/* Copyright 2016-2021 Alexander Grund
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

#include <pmacc/boost_workaround.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch.hpp>


#if TEST_DIM == 2
using pmacc::test::PMaccFixture2D;
static PMaccFixture2D fixture;
#else
using pmacc::test::PMaccFixture3D;
static PMaccFixture3D fixture;
#endif

#include "IdProvider.hpp"
#include "memory/SuperCell.hpp"
