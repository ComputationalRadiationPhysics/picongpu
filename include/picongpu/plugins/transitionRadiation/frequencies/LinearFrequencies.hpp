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
            namespace linearFrequencies
            {
                class FreqFunctor
                {
                public:
                    FreqFunctor(void) = default;

                    HDINLINE float_X operator()(int const ID)
                    {
                        return omegaMin + float_X(ID) * deltaOmega;
                    }

                    HINLINE float_X get(int const ID)
                    {
                        return operator()(ID);
                    }
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
                    std::string params = std::string("lin\t");
                    params += std::to_string(nOmega) + "\t";
                    params += std::to_string(SI::omegaMin) + "\t";
                    params += std::to_string(SI::omegaMax) + "\t";
                    return params;
                }

            } // namespace linearFrequencies
        } // namespace transitionRadiation
    } // namespace plugins
} // namespace picongpu
