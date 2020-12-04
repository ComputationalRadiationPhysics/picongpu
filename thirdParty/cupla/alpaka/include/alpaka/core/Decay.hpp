/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <type_traits>

//-----------------------------------------------------------------------------
//! Wrapper around std::decay_t for parameter pack expansion expressions
//
// Works around Intel compiler internal error when used in empty template pack
// extension as discussed in #995. It seems not possible to make a workaround
// with pure C++ tools, like an alias template, so macro it is. Note that
// there is no known issue outside of empty parameter pack expansions,
// so the normal std::decay_t can and should be used there.
//
// The choice of macro over writing typename std::decay<Type>::type explicitly
// in parameter pack expansion expressions is to avoid warnings from diagnostic
// tools, and also for brevity.
//-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL || BOOST_COMP_PGI
#    define ALPAKA_DECAY_T(Type) typename std::decay<Type>::type
#else
#    define ALPAKA_DECAY_T(Type) std::decay_t<Type>
#endif
