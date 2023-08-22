/* Copyright 2022 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

#include <type_traits>

//! Wrapper around std::decay_t for parameter pack expansion expressions
//
// Works around PGI compiler internal error when used in empty template pack
// extension as discussed in #995. It seems not possible to make a workaround
// with pure C++ tools, like an alias template, so macro it is. Note that
// there is no known issue outside of empty parameter pack expansions,
// so the normal std::decay_t can and should be used there.
//
// The choice of macro over writing typename std::decay<Type>::type explicitly
// in parameter pack expansion expressions is to avoid warnings from diagnostic
// tools, and also for brevity.
#if BOOST_COMP_PGI
#    define ALPAKA_DECAY_T(Type) typename std::decay<Type>::type
#else
#    define ALPAKA_DECAY_T(Type) std::decay_t<Type>
#endif

namespace alpaka
{
    //! Provides a decaying wrapper around std::is_same. Example: is_decayed_v<volatile float, float> returns true.
    template<typename T, typename U>
    inline constexpr auto is_decayed_v = std::is_same_v<ALPAKA_DECAY_T(T), ALPAKA_DECAY_T(U)>;
} // namespace alpaka
