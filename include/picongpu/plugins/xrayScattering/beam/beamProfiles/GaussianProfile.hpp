/* Copyright 2020-2021 Pawel Ordyna
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
        namespace xrayScattering
        {
            namespace beam
            {
                namespace beamProfiles
                {
                    /** Gaussian beam transverse profile.
                     *
                     * @tparam T_ParamClass Param Class defining @f[ /sigma_x / /simga_y @f] .
                     */
                    template<typename T_ParamClass>
                    struct GaussianProfile : public T_ParamClass
                    {
                        using ParamClass = T_ParamClass;

                        static HDINLINE float_X getFactor(float_X const& x, float_X const& y)
                        {
                            constexpr float_X s_x = ParamClass::sigmaX_SI / UNIT_LENGTH;
                            constexpr float_X s_y = ParamClass::sigmaY_SI / UNIT_LENGTH;
                            constexpr float_X tmp_x = x / s_x;
                            constexpr float_X tmp_y = y / s_y;
                            float_X exponent = -0.5 * (tmp_x * tmp_x + tmp_y * tmp_y);
                            return math::exp(exponent);
                        }
                    };
                } // namespace beamProfiles
            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
