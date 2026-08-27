/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>

namespace alpaka
{
    namespace mem_order
    {

        /**
         * The user requested memory order may be converted to a stronger memory order guarantee if the backend does
         * not support the requested memory ordering
         * If the user requests a memory ordering which is stronger than what is possible, we throw an error statically
         */

        struct MemoryOrderTag
        {
        };

        struct SeqCst : MemoryOrderTag
        {
        };

        struct AcqRel : MemoryOrderTag
        {
        };

        struct Release : MemoryOrderTag
        {
        };

        struct Acquire : MemoryOrderTag
        {
        };

        struct Relaxed : MemoryOrderTag
        {
        };

        static constexpr SeqCst seq_cst;
        static constexpr AcqRel acq_rel;
        static constexpr Release release;
        static constexpr Acquire acquire;
        static constexpr Relaxed relaxed;

    } // namespace mem_order

    template<typename T>
    concept MemoryOrder = std::derived_from<T, mem_order::MemoryOrderTag>;

} // namespace alpaka
