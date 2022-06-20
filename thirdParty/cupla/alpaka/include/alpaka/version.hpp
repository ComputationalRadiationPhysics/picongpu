/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/predef/version_number.h>

#define ALPAKA_VERSION_MAJOR 1
#define ALPAKA_VERSION_MINOR 0
#define ALPAKA_VERSION_PATCH 0

//! The alpaka library version number
#define ALPAKA_VERSION BOOST_VERSION_NUMBER(ALPAKA_VERSION_MAJOR, ALPAKA_VERSION_MINOR, ALPAKA_VERSION_PATCH)
