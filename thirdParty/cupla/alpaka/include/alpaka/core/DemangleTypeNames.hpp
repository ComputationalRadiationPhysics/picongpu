/* Copyright 2022 Andrea Bocci, Antonio Di Pilato
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <boost/core/demangle.hpp>

namespace alpaka::core
{
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#    pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#endif
    template<typename T>
    inline const std::string demangled = boost::core::demangle(typeid(T).name());
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
} // namespace alpaka::core
