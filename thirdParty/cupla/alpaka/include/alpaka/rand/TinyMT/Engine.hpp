/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/TinyMT/tinymt32.h>

#include <cstdint>

namespace alpaka::rand::engine::cpu
{
    //! Implementation of std::UniformRandomBitGenerator for TinyMT32
    struct TinyMTengine
    {
        using result_type = std::uint32_t;

        static constexpr auto default_seed() -> result_type
        {
            return 42u;
        }

        void seed(result_type value = default_seed())
        {
            // parameters from TinyMT/jump/sample.c
            prng.mat1 = 0x8f7011ee;
            prng.mat2 = 0xfc78ff1f;
            prng.tmat = 0x3793fdff;

            tinymt32_init(&prng, value);
        }

        TinyMTengine(std::uint32_t const& seedValue)
        {
            seed(seedValue);
        }

        TinyMTengine()
        {
            seed(default_seed());
        }

        auto operator()() -> result_type
        {
            return tinymt32_generate_uint32(&prng);
        }

        static constexpr auto min() -> result_type
        {
            return 0u;
        }

        static constexpr auto max() -> result_type
        {
            return UINT32_MAX;
        }

        void discard(unsigned long long) // z
        {
            // not implemented
            // tinymt32_jump( &prng, z, z );
        }

        tinymt32_t prng;
    };
} // namespace alpaka::rand::engine::cpu
