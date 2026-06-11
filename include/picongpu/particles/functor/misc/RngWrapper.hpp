/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
                 * @tparam T_Worker type of the alpaka accelerator
                 * @tparam T_Rng type of the random number generator
                 */
                template<typename T_Worker, typename T_Rng>
                struct RngWrapper
                {
                    DINLINE RngWrapper(
                        T_Worker const& worker,
                        T_Rng const& rng

                        )
                        : m_worker(&worker)
                        , m_rng(rng)
                    {
                    }

                    //! generate a random number
                    DINLINE
                    typename T_Rng::result_type operator()() const
                    {
                        return m_rng(*m_worker);
                    }

                    T_Worker const* m_worker;
                    mutable T_Rng m_rng;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu
