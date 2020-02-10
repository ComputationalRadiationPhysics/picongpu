/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <cstdint>


namespace picongpu
{
namespace fields
{
namespace maxwellSolver
{

    /** Forward declaration to avoid mutual including with YeePML.hpp
     *
     * @tparam T_CurrentInterpolation current interpolation functor
     * @tparam T_CurlE functor to compute curl of E
     * @tparam T_CurlB functor to compute curl of B
     */
    template<
        typename T_CurrentInterpolation,
        typename T_CurlE,
        typename T_CurlB
    >
    class YeePML;

} // namespace maxwellSolver

namespace absorber
{
namespace detail
{

    /** Number of cells helper
     *
     * The general version uses exponential absorber settings
     *
     * @tparam T_FieldSolver field solver
     */
    template< typename T_FieldSolver >
    struct NumCellsHelper
    {
        static constexpr uint32_t xNegative = ABSORBER_CELLS[ 0 ][ 0 ];
        static constexpr uint32_t xPositive = ABSORBER_CELLS[ 0 ][ 1 ];
        static constexpr uint32_t yNegative = ABSORBER_CELLS[ 1 ][ 0 ];
        static constexpr uint32_t yPositive = ABSORBER_CELLS[ 1 ][ 1 ];
        static constexpr uint32_t zNegative = ABSORBER_CELLS[ 2 ][ 0 ];
        static constexpr uint32_t zPositive = ABSORBER_CELLS[ 2 ][ 1 ];
    };

    namespace pml = maxwellSolver::yeePML;

    /** Number of cells helper
     *
     * Specialization for PML
     *
     * @tparam T_CurrentInterpolation current interpolation for YeePML
     * @tparam T_CurlE curl E for YeePML
     * @tparam T_CurlB curl B for YeePML
     */
    template<
        typename T_CurrentInterpolation,
        typename T_CurlE,
        typename T_CurlB
    >
    struct NumCellsHelper<
        maxwellSolver::YeePML<
            T_CurrentInterpolation,
            T_CurlE,
            T_CurlB
        >
    >
    {
        static constexpr uint32_t xNegative = pml::NUM_CELLS[ 0 ][ 0 ];
        static constexpr uint32_t xPositive = pml::NUM_CELLS[ 0 ][ 1 ];
        static constexpr uint32_t yNegative = pml::NUM_CELLS[ 1 ][ 0 ];
        static constexpr uint32_t yPositive = pml::NUM_CELLS[ 1 ][ 1 ];
        static constexpr uint32_t zNegative = pml::NUM_CELLS[ 2 ][ 0 ];
        static constexpr uint32_t zPositive = pml::NUM_CELLS[ 2 ][ 1 ];
    };

    //! Number of cells helper for the selected field solver + absorber
    using NumCells = NumCellsHelper< Solver >;

} // namespace detail

    /** Number of absorber cells along each boundary
     *
     * Is uniform for both PML and exponential damping absorbers.
     * First index: 0 = x, 1 = y, 2 = z.
     * Second index: 0 = negative (min coordinate), 1 = positive (max coordinate).
     * Not for ODR-use.
     */
    constexpr uint32_t numCells[ 3 ][ 2 ] = {
        { detail::NumCells::xNegative, detail::NumCells::xPositive },
        { detail::NumCells::yNegative, detail::NumCells::yPositive },
        { detail::NumCells::zNegative, detail::NumCells::zPositive }
    };

} // namespace absorber
} // namespace fields
} // namespace picongpu
