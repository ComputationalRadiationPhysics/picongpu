/* Copyright 2024 Andrea Bocci, Felice Pantaleo
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"

// clang 9/10/11 together with nvcc<11.6.0 as host compiler fails at compile time when using boost::atomic_ref
#ifdef BOOST_COMP_CLANG_AVAILABLE
#    if(BOOST_COMP_CLANG < BOOST_VERSION_NUMBER(12, 0, 0) && BOOST_COMP_NVCC                                          \
        && BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(11, 6, 0))
#        if !defined(ALPAKA_DISABLE_ATOMIC_ATOMICREF)
#            define ALPAKA_DISABLE_ATOMIC_ATOMICREF
#        endif
#    endif
#endif // BOOST_COMP_CLANG_AVAILABLE

#include "alpaka/atomic/AtomicAtomicRef.hpp"
#include "alpaka/atomic/AtomicStdLibLock.hpp"

namespace alpaka
{
#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
    using AtomicCpu = AtomicAtomicRef;
#else
    using AtomicCpu = AtomicStdLibLock<16>;
#endif // ALPAKA_DISABLE_ATOMIC_ATOMICREF

} // namespace alpaka
