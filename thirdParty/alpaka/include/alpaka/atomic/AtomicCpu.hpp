/* Copyright 2025 Andrea Bocci, Felice Pantaleo
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)                    \
    || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)                   \
    || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)

#    include "alpaka/atomic/AtomicAtomicRef.hpp"
#    include "alpaka/atomic/AtomicStdLibLock.hpp"

namespace alpaka
{
#    ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
    using AtomicCpu = AtomicAtomicRef;
#    else
    using AtomicCpu = AtomicStdLibLock<16>;
#    endif // ALPAKA_DISABLE_ATOMIC_ATOMICREF

} // namespace alpaka

#endif
