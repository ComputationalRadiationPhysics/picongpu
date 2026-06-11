/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/test/PMaccFixture.hpp>

#include <catch2/catch_test_macros.hpp>


#if TEST_DIM == 2
using pmacc::test::PMaccFixture2D;
static PMaccFixture2D fixture;
#else
using pmacc::test::PMaccFixture3D;
static PMaccFixture3D fixture;
#endif

#include "IdProvider.hpp"
#include "memory/SuperCell.hpp"
