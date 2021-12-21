/* Copyright 2015-2021 Axel Huebl, Sergei Bastrakov
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

#include "picongpu/fields/currentInterpolation/Binomial.hpp"
#include "picongpu/fields/currentInterpolation/None.hpp"

#include <pmacc/math/Vector.hpp>
#include <pmacc/traits/GetStringProperties.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace currentInterpolation
        {
            /** Singleton to represent current interpolation kind
             *
             * It does not perform interpolation itself, that is done by functors None and Binomial.
             * Provides run-time utilities to get margin values and string properties.
             */
            struct CurrentInterpolation
            {
            public:
                //! Supported interpolation kinds
                enum class Kind
                {
                    None,
                    Binomial
                };

                //! Interpolation kind used in the simulation
                Kind kind = Kind::Binomial;

                //! Get the single instance of the current interpolation object
                static CurrentInterpolation& get()
                {
                    static CurrentInterpolation instance;
                    return instance;
                }

                //! Get string properties
                static pmacc::traits::StringProperty getStringProperties()
                {
                    return get().kind == Kind::None ? None::getStringProperties() : Binomial::getStringProperties();
                }

                //! Get the lower margin of the used interpolation functor
                static pmacc::math::Vector<int, simDim> getLowerMargin()
                {
                    return get().kind == Kind::None ? None::LowerMargin::toRT() : Binomial::LowerMargin::toRT();
                }

                //! Get the upper margin of the used interpolation functor
                static pmacc::math::Vector<int, simDim> getUpperMargin()
                {
                    return get().kind == Kind::None ? None::UpperMargin::toRT() : Binomial::UpperMargin::toRT();
                }

                //! Copy construction is forbidden
                CurrentInterpolation(CurrentInterpolation const&) = delete;

                //! Assignment is forbidden
                CurrentInterpolation& operator=(CurrentInterpolation const&) = delete;

            private:
                CurrentInterpolation() = default;
                ~CurrentInterpolation() = default;
            };

        } // namespace currentInterpolation
    } // namespace fields
} // namespace picongpu
