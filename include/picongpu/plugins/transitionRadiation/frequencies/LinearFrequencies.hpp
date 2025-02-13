/* Copyright 2013-2024 Heiko Burau, Rene Widera, Richard Pausch, Finn-Ole Carstens
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
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
            namespace linear_frequencies
            {
                class FreqFunctor
                {
                public:
                    FreqFunctor(void) = default;

                    HDINLINE float_X operator()(const int ID)
                    {
                        return omega_min + float_X(ID) * deltaOmega;
                    }

                    HINLINE float_X get(const int ID)
                    {
                        return operator()(ID);
                    }
                }; // FreqFunctor

                class InitFreqFunctor
                {
                public:
                    InitFreqFunctor(void) = default;

                    HINLINE void Init(const std::string path)
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
                    params += std::to_string(N_omega) + "\t";
                    params += std::to_string(SI::omega_min) + "\t";
                    params += std::to_string(SI::omega_max) + "\t";
                    return params;
                }

            } // namespace linear_frequencies
        } // namespace transitionRadiation
    } // namespace plugins
} // namespace picongpu
