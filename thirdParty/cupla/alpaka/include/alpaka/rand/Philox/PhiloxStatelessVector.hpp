/* Copyright 2022 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/rand/Philox/PhiloxBaseTraits.hpp>

#include <utility>


namespace alpaka::rand::engine
{
    /** Philox-stateless engine generating a vector of numbers
     *
     * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
     * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
     * themselves, but the internal state comprises only of a single counter and a key.
     *
     * @tparam TAcc Accelerator type as defined in alpaka/acc
     * @tparam TParams Basic parameters for the Philox algorithm
     */
    template<typename TAcc, typename TParams>
    class PhiloxStatelessVector : public trait::PhiloxStatelessBaseTraits<TAcc, TParams>::Base
    {
    };
} // namespace alpaka::rand::engine
