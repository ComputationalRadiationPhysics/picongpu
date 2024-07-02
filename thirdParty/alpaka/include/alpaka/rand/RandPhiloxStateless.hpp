/* Copyright 2022 Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/Philox/PhiloxStateless.hpp"
#include "alpaka/rand/Traits.hpp"

namespace alpaka::rand
{
    /** Most common Philox engine variant, stateless, outputs a 4-vector of floats
     *
     * This is a variant of the Philox engine generator which outputs a vector containing 4 floats. The counter
     * size is \f$4 \times 32 = 128\f$ bits. Since the engine returns the whole generated vector, it is up to the
     * user to extract individual floats as they need. The benefit is smaller state size since the state does not
     * contain the intermediate results. The total size of the state is 192 bits = 24 bytes.
     *
     * Ref.: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and
     * Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     */
    class PhiloxStateless4x32x10Vector
        : public alpaka::rand::engine::PhiloxStateless<engine::PhiloxParams<4, 32, 10>>
        , public concepts::Implements<ConceptRand, PhiloxStateless4x32x10Vector>
    {
    public:
        using EngineParams = engine::PhiloxParams<4, 32, 10>;
    };
} // namespace alpaka::rand
