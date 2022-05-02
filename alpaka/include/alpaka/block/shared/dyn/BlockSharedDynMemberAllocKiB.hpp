/* Copyright 2022 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
