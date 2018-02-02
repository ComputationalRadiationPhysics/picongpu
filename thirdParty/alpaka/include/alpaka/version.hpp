/**
* \file
* Copyright 2016-2017 Erik Zenker, Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/predef/version_number.h>

#define ALPAKA_VERSION_MAJOR 0
#define ALPAKA_VERSION_MINOR 2
#define ALPAKA_VERSION_PATCH 0

//! The alpaka library version number
#define ALPAKA_VERSION BOOST_VERSION_NUMBER(ALPAKA_VERSION_MAJOR, ALPAKA_VERSION_MINOR, ALPAKA_VERSION_PATCH)
