/* Copyright 2018-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/particles/memory/dataTypes/SuperCell.hpp>


namespace pmacc
{
    namespace test
    {
        namespace particles
        {
            namespace memory
            {
                template<typename T_SuperCell>
                struct TestNumParticlesLastFrame
                {
                    struct FrameTypeDummy
                    {
                        using SuperCellSize = T_SuperCell;
                    };

                    /** test a combination
                     *
                     * @param numParticlesPerCell number of particles within the test supercell
                     * @param particleLastFrame the assumed result with the given number of particles
                     *                          and T_SuperCell
                     */
                    HINLINE void operator()(uint32_t numParticlesPerCell, uint32_t particleLastFrame)
                    {
                        pmacc::SuperCell<FrameTypeDummy> superCell;
                        superCell.setNumParticles(numParticlesPerCell);

                        REQUIRE(superCell.getSizeLastFrame() == particleLastFrame);
                    }
                };

            } // namespace memory
        } // namespace particles
    } // namespace test
} // namespace pmacc

/* The supercell test is always performed with a 3 dimensional supercell
 * because the supercell is agnostic about the number of dimensions.
 */
TEST_CASE("particles::SuperCell", "[SuperCell]")
{
    using namespace pmacc::test::particles::memory;
    TestNumParticlesLastFrame<pmacc::math::CT::Int<8, 8, 4>> cell256{};

    // no particles in the supercell
    cell256(0u, 0u);
    // one full frame
    cell256(256u, 256u);
    // two full frames
    cell256(512u, 256u);
    // edge cases
    cell256(255u, 255u);
    cell256(257u, 1u);
    cell256(1u, 1u);

    using namespace pmacc::test::particles::memory;
    TestNumParticlesLastFrame<pmacc::math::CT::Int<3, 3, 3>> cell27{};

    // no particles in the supercell
    cell27(0u, 0u);
    // one full frame
    cell27(27u, 27u);
    // two full frames
    cell27(54u, 27u);
    // edge cases
    cell27(26u, 26u);
    cell27(28u, 1u);
    cell27(1u, 1u);
}
