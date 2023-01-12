/* Copyright 2021 Andrea Bocci, Felice Pantaleo
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/version.hpp>

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    define ALPAKA_DISABLE_ATOMIC_ATOMICREF
#endif

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    include <alpaka/atomic/AtomicAtomicRef.hpp>
#else
#    include <alpaka/atomic/AtomicStdLibLock.hpp>
#endif // ALPAKA_DISABLE_ATOMIC_ATOMICREF

namespace alpaka
{
#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
    using AtomicCpu = AtomicAtomicRef;
#else
    using AtomicCpu = AtomicStdLibLock<16>;
#endif // ALPAKA_DISABLE_ATOMIC_ATOMICREF

} // namespace alpaka
