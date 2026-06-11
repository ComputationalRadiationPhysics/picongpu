/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/radiation/param.hpp"

#include <pmacc/math/math.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            namespace log_frequencies
            {
                class FreqFunctor
                {
                public:
                    FreqFunctor(void)
                    {
                        omega_log_min = math::log(omega_min);
                        delta_omega_log = (math::log(omega_max) - omega_log_min) / float_X(N_omega - 1);
                    }

                    HDINLINE float_X operator()(int const ID)
                    {
                        return math::exp(omega_log_min + (float_X(ID)) * delta_omega_log);
                    }

                    HINLINE float_X get(int const ID)
                    {
                        return operator()(ID);
                    }

                private:
                    float_X omega_log_min;
                    float_X delta_omega_log;
                };

                class InitFreqFunctor
                {
                public:
                    InitFreqFunctor(void) = default;

                    HINLINE void Init(std::string const path)
                    {
                    }

                    HINLINE FreqFunctor getFunctor(void)
                    {
                        return {};
                    }
                };


            } // namespace log_frequencies
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
