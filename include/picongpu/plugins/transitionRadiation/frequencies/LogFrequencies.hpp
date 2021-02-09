/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Finn-Ole Carstens
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

                    HDINLINE float_X operator()(const int ID)
                    {
                        return math::exp(omega_log_min + (float_X(ID)) * delta_omega_log);
                    }

                    HINLINE float_X get(const int ID)
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
                    InitFreqFunctor(void)
                    {
                    }

                    HINLINE void Init(const std::string path)
                    {
                    }


                    HINLINE FreqFunctor getFunctor(void)
                    {
                        return FreqFunctor();
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
