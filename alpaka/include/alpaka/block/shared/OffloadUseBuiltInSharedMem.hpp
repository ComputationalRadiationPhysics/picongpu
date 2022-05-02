/* Copyright 2022 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    struct OffloadBuiltInSharedMemOff
    {
    };
    struct OffloadBuiltInSharedMemFixed
    {
    };
    struct OffloadBuiltInSharedMemAlloc
    {
    };
#ifdef ALPAKA_OFFLOAD_BUILTIN_SHARED_MEM_FIXED
    using OffloadBuiltInSharedMem = OffloadBuiltInSharedMemFixed;
#elif defined(ALPAKA_OFFLOAD_BUILTIN_SHARED_MEM_ALLOC)
    using OffloadBuiltInSharedMem = OffloadBuiltInSharedMemAlloc;
#else
    using OffloadBuiltInSharedMem = OffloadBuiltInSharedMemOff;
#endif
} // namespace alpaka
