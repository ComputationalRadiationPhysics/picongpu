/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/radiation/param.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            namespace linear_frequencies
            {
                class FreqFunctor
                {
                public:
                    FreqFunctor(void) = default;

                    HDINLINE float_X operator()(int const ID)
                    {
                        return omega_min + float_X(ID) * delta_omega;
                    }

                    HINLINE float_X get(int const ID)
                    {
                        return operator()(ID);
                    }
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

            } // namespace linear_frequencies
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
