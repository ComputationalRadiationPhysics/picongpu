/* Copyright 2019-2021 Sergei Bastrakov
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

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/algorithms/math/floatMath/floatingPoint.tpp>

#include <cstdint>
#include <stdexcept>
#include <string>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace yeePML
            {
                /** Parameters of PML, except thickness
                 *
                 * A detailed description and recommended ranges are given in pml.param,
                 * normalizations and unit conversions in pml.unitless.
                 */
                struct Parameters
                {
                    /** Max value of artificial electric conductivity
                     *
                     * Components correspond to directions. Normalized, so that
                     * normalizedSigma = sigma / eps0 = sigma* / mue0.
                     * Unit: 1/unit_time in PIC units
                     */
                    floatD_X normalizedSigmaMax;

                    /** Order of polynomial growth of sigma and kappa
                     *
                     * The growth is from PML internal boundary to the external boundary.
                     * Sigma grows from 0, kappa from 1, both to their max values.
                     */
                    float_X sigmaKappaGradingOrder;

                    /** Max value of coordinate stretching coefficient
                     *
                     * Unitless.
                     */
                    floatD_X kappaMax;

                    /** Max value of complex frequency shift
                     *
                     * Components correspond to directions. Normalized by eps0.
                     * Unit: 1/unit_time in PIC units
                     */
                    floatD_X normalizedAlphaMax;

                    /** Order of polynomial growth of alpha
                     *
                     * The growth is from PML external boundary to the internal boundary.
                     * Grows from 0 to the max value.
                     */
                    float_X alphaGradingOrder;
                };

                //! Thickness of PML at each border, in number of cells
                struct Thickness
                {
                    //! Negative border is at the local domain sides minimum in coordinates
                    DataSpace<simDim> negativeBorder;
                    //! Positive border is at the local domain sides maximum in coordinates
                    DataSpace<simDim> positiveBorder;

                    /** Element access with indexing used in the .param file
                     *
                     * This is only for initialization convenience and so does not have
                     * a device version. Since this is not performance-critical at all,
                     * do range checks on parameters.
                     *
                     * @param axis 0 = x, 1 = y, 2 = z
                     * @param direction 0 = negative, 1 = positive
                     */
                    int& operator()(uint32_t const axis, uint32_t const direction)
                    {
                        if(axis >= simDim)
                            throw std::out_of_range(
                                "In Thickness::operator() the axis = " + std::to_string(axis) + " is invalid");
                        if(direction == 0)
                            return negativeBorder[axis];
                        else if(direction == 1)
                            return positiveBorder[axis];
                        else
                            throw std::out_of_range(
                                "In Thickness::operator() the direction = " + std::to_string(direction)
                                + " is invalid");
                    }
                };

            } // namespace yeePML
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
