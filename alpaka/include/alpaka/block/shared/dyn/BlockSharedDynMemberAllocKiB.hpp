/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace alpaka
{
#ifndef ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB
#    define ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB 47u
#endif
    constexpr std::uint32_t BlockSharedDynMemberAllocKiB = ALPAKA_BLOCK_SHARED_DYN_MEMBER_ALLOC_KIB;
} // namespace alpaka
