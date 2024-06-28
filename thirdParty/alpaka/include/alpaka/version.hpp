/* Copyright 2022 Benjamin Worpitz, Erik Zenker, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <boost/predef/version_number.h>

#define ALPAKA_VERSION_MAJOR 1
#define ALPAKA_VERSION_MINOR 2
#define ALPAKA_VERSION_PATCH 0

//! The alpaka library version number
#define ALPAKA_VERSION BOOST_VERSION_NUMBER(ALPAKA_VERSION_MAJOR, ALPAKA_VERSION_MINOR, ALPAKA_VERSION_PATCH)
