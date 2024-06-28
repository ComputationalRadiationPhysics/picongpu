/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/PhiloxStateless.hpp"

namespace alpaka::rand::engine
{
    /** Common class for Philox family engines
     *
     * Checks the validity of passed-in parameters and calls the backend methods to perform N rounds of the
     * Philox shuffle.
     *
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     */
    template<typename TParams>
    struct PhiloxStatelessKeyedBase : public PhiloxStateless<TParams>
    {
    public:
        using Counter = typename PhiloxStateless<TParams>::Counter;
        using Key = typename PhiloxStateless<TParams>::Key;

        Key const m_key;

        PhiloxStatelessKeyedBase(Key&& key) : m_key(std::move(key))
        {
        }

        ALPAKA_FN_HOST_ACC auto operator()(Counter const& counter) const
        {
            return this->generate(counter, m_key);
        }
    };
} // namespace alpaka::rand::engine
