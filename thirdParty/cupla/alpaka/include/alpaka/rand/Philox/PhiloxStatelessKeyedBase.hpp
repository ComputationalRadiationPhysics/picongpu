/* Copyright 2022 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxStateless.hpp>

namespace alpaka::rand::engine
{
    /** Common class for Philox family engines
     *
     * Checks the validity of passed-in parameters and calls the \a TBackend methods to perform N rounds of the
     * Philox shuffle.
     *
     * @tparam TBackend device-dependent backend, specifies the array types
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     */
    template<typename TBackend, typename TParams>
    struct PhiloxStatelessKeyedBase : public PhiloxStateless<TBackend, TParams>
    {
    public:
        using Counter = typename PhiloxStateless<TBackend, TParams>::Counter;
        using Key = typename PhiloxStateless<TBackend, TParams>::Key;

        const Key m_key;

        PhiloxStatelessKeyedBase(Key&& key) : m_key(std::move(key))
        {
        }

        ALPAKA_FN_HOST_ACC auto operator()(Counter const& counter) const
        {
            return this->generate(counter, m_key);
        }
    };
} // namespace alpaka::rand::engine
