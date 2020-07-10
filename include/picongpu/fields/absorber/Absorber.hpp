/* Copyright 2013-2020 Axel Huebl, Rene Widera, Sergei Bastrakov, Klaus Steiniger
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

#include <pmacc/Environment.hpp>
#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>
#include <sstream>
#include <string>


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

    //! Forward declaration to avoid mutual including with ExponentialDamping.hpp
    class ExponentialDamping;

namespace detail
{

    /** Get string properties of the absorber
     *
     * @param name absorber name
     */
    HINLINE pmacc::traits::StringProperty getStringProperties(
        std::string const & name
    );

    /** Absorber wrapper
     *
     * Provides unified interface for the absorber information:
     * size along the 6 boundaries and getStringProperties() implementation.
     * Currently does not provide the computational part, only description.
     *
     * The general version uses exponential absorber settings since this is the
     * default absorber.
     *
     * @tparam T_FieldSolver field solver
     */
    template< typename T_FieldSolver >
    struct Absorber
    {
        //! Number of absorber cells along the min x boundary
        static constexpr uint32_t xNegativeNumCells = ABSORBER_CELLS[ 0 ][ 0 ];

        //! Number of absorber cells along the max x boundary
        static constexpr uint32_t xPositiveNumCells = ABSORBER_CELLS[ 0 ][ 1 ];

        //! Number of absorber cells along the min y boundary
        static constexpr uint32_t yNegativeNumCells = ABSORBER_CELLS[ 1 ][ 0 ];

        //! Number of absorber cells along the max y boundary
        static constexpr uint32_t yPositiveNumCells = ABSORBER_CELLS[ 1 ][ 1 ];

        //! Number of absorber cells along the min z boundary
        static constexpr uint32_t zNegativeNumCells = ABSORBER_CELLS[ 2 ][ 0 ];

        //! Number of cells along the max z boundary
        static constexpr uint32_t zPositiveNumCells = ABSORBER_CELLS[ 2 ][ 1 ];

        //! Get string properties of the absorber
        static pmacc::traits::StringProperty getStringProperties()
        {
            return detail::getStringProperties( "exponential damping" );
        }
    };

    namespace pml = maxwellSolver::Pml;

    /** Absorber wrapper
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
    struct Absorber<
        maxwellSolver::YeePML<
            T_CurrentInterpolation,
            T_CurlE,
            T_CurlB
        >
    >
    {
        //! Number of absorber cells along the min x boundary
        static constexpr uint32_t xNegativeNumCells = pml::NUM_CELLS[ 0 ][ 0 ];

        //! Number of absorber cells along the max x boundary
        static constexpr uint32_t xPositiveNumCells = pml::NUM_CELLS[ 0 ][ 1 ];

        //! Number of absorber cells along the min y boundary
        static constexpr uint32_t yNegativeNumCells = pml::NUM_CELLS[ 1 ][ 0 ];

        //! Number of absorber cells along the max y boundary
        static constexpr uint32_t yPositiveNumCells = pml::NUM_CELLS[ 1 ][ 1 ];

        //! Number of absorber cells along the min z boundary
        static constexpr uint32_t zNegativeNumCells = pml::NUM_CELLS[ 2 ][ 0 ];

        //! Number of absorber cells along the max z boundary
        static constexpr uint32_t zPositiveNumCells = pml::NUM_CELLS[ 2 ][ 1 ];

        //! Get string properties of the absorber
        static pmacc::traits::StringProperty getStringProperties()
        {
            return detail::getStringProperties( "convolutional PML" );
        }

    };

} // namespace detail

    /** Absorber description implementing getStringProperties()
     *
     * To be used for writing absorber meta information, does not provide
     * interface for running the absorber
     */
    using Absorber = detail::Absorber< Solver >;

    /** Number of absorber cells along each boundary
     *
     * Is uniform for both PML and exponential damping absorbers.
     * First index: 0 = x, 1 = y, 2 = z.
     * Second index: 0 = negative (min coordinate), 1 = positive (max coordinate).
     * Not for ODR-use.
     */
    constexpr uint32_t numCells[ 3 ][ 2 ] = {
        { Absorber::xNegativeNumCells, Absorber::xPositiveNumCells },
        { Absorber::yNegativeNumCells, Absorber::yPositiveNumCells },
        { Absorber::zNegativeNumCells, Absorber::zPositiveNumCells }
    };

namespace detail
{

    // Implementation has to be after numCells is defined
    pmacc::traits::StringProperty getStringProperties( std::string const & name )
    {
        pmacc::traits::StringProperty propList;
        const DataSpace<DIM3> periodic =
            Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

        for( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
        {
            // for each planar direction: left right top bottom back front
            if( FRONT % i == 0 )
            {
                const std::string directionName = ExchangeTypeNames()[i];
                const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);

                bool isPeriodic = false;
                uint32_t axis = 0;    // x(0) y(1) z(2)
                uint32_t axisDir = 0; // negative (0), positive (1)
                for( uint32_t d = 0; d < simDim; d++ )
                {
                    if( relDir[d] * periodic[d] != 0 )
                        isPeriodic = true;
                    if( relDir[d] != 0 )
                        axis = d;
                }
                if( relDir[axis] > 0 )
                    axisDir = 1;

                std::string boundaryName = "open"; // absorbing boundary
                if( isPeriodic )
                    boundaryName = "periodic";

                if( boundaryName == "open" )
                {
                    std::ostringstream boundaryParam;
                    boundaryParam << name + " over "
                        << numCells[axis][axisDir] << " cells";
                    propList[directionName]["param"] = boundaryParam.str();
                }
                else
                {
                    propList[directionName]["param"] = "none";
                }

                propList[directionName]["name"] = boundaryName;
            }
        }
        return propList;
    }

} // namespace detail

} // namespace absorber
} // namespace fields
} // namespace picongpu
