/* Copyright 2017-2021 Rene Widera
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

#include <pmacc/types.hpp>

#include <functional>


namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            namespace misc
            {
                /** wraps an random number generator together with an alpaka accelerator
                 *
                 * This class allows to generate random numbers without passing the accelerator
                 * to each functor call.
                 *
                 * @tparam T_Acc type of the alpaka accelerator
                 * @tparam T_Rng type of the random number generator
                 */
                template<typename T_Acc, typename T_Rng>
                struct RngWrapper
                {
                    DINLINE RngWrapper(
                        T_Acc const& acc,
                        T_Rng const& rng

                        )
                        : m_acc(&acc)
                        , m_rng(rng)
                    {
                    }

                    //! generate a random number
                    DINLINE
                    typename T_Rng::result_type operator()()
                    {
                        return m_rng(*m_acc);
                    }

                    T_Acc const* m_acc;
                    mutable T_Rng m_rng;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
