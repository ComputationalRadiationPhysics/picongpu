/* Copyright 2022 Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/mem/fence/Traits.hpp>

namespace alpaka
{
    //! The OpenACC memory fence.
    class MemFenceOacc : public concepts::Implements<ConceptMemFence, MemFenceOacc>
    {
    };

    namespace trait
    {
        template<typename TMemScope>
        struct MemFence<MemFenceOacc, TMemScope>
        {
            static auto mem_fence(MemFenceOacc const&, TMemScope const&)
            {
                /* Memory fences are by design not available in OpenACC (see OpenACC 3.1 spec, section 1.3).
                 * Quote from the spec:
                 *
                 * Some accelerators implement a weak memory model. In particular, they do not support memory
                 * coherence between operations executed by different threads; even on the same execution unit, mem-
                 * ory coherence is only guaranteed when the memory operations are separated by an explicit memory
                 * fence. Otherwise, if one thread updates a memory location and another reads the same location, or
                 * two threads store a value to the same location, the hardware may not guarantee the same result for
                 * each execution. While a compiler can detect some potential errors of this nature, it is nonetheless
                 * possible to write a compute region that produces inconsistent numerical results. (End of quote)
                 *
                 * As noted in the OpenACC 3.1 spec, section 1.2, programmers should not attempt to implement any form
                 * of explicit synchronization themselves.
                 *
                 * We therefore do not implement a memory fence for OpenACC and instead throw a compile-time error
                 * which informs the user to use another back-end.
                 */
                static_assert(!sizeof(TMemScope), "Memory fences are not available in the OpenACC back-end");
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
