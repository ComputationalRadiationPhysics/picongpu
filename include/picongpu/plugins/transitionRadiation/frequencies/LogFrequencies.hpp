/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Finn-Ole Carstens
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/transitionRadiation/param.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace transitionRadiation
        {
            namespace logFrequencies
            {
                class FreqFunctor
                {
                public:
                    FreqFunctor(void)
                    {
                        omega_log_min = math::log(omegaMin);
                        delta_omega_log = (math::log(omegaMax) - omega_log_min) / float_X(nOmega - 1);
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
                }; // FreqFunctor

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
                }; // InitFreqFunctor

                //! @return frequency params as string
                HINLINE
                std::string getParameters(void)
                {
                    std::string params = std::string("log\t");
                    params += std::to_string(nOmega) + "\t";
                    params += std::to_string(SI::omegaMin) + "\t";
                    params += std::to_string(SI::omegaMax) + "\t";
                    return params;
                }

            } // namespace logFrequencies
        } // namespace transitionRadiation
    } // namespace plugins
} // namespace picongpu
