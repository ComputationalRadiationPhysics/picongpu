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
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/Complex.hpp>
#include <pmacc/memory/buffers/Buffer.hpp>

#include <algorithm>
#include <vector>

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            template<typename T>
            std::vector<T> extractReal(Buffer<pmacc::math::Complex<T>, DIM1>& complexBuffer)
            {
                std::vector<T> realValues;
                auto size = complexBuffer.getCurrentSize();
                auto dataBox = complexBuffer.getDataBox();
                realValues.reserve(size);
                for(uint32_t ii = 0; ii < size; ii++)
                {
                    realValues.push_back(dataBox[ii].get_real());
                }
                return realValues;
            }

            template<typename T>
            std::vector<T> extractImag(Buffer<pmacc::math::Complex<T>, DIM1>& complexBuffer)
            {
                std::vector<T> imagValues;
                auto size = complexBuffer.getCurrentSize();
                auto dataBox = complexBuffer.getDataBox();
                imagValues.reserve(size);
                for(uint32_t ii = 0; ii < size; ii++)
                {
                    imagValues.push_back(dataBox[ii].get_imag());
                }
                return imagValues;
            }

            template<typename T>
            std::vector<T> extractReal(std::vector<pmacc::math::Complex<T>> const& complexVec)
            {
                std::vector<T> realValues;
                realValues.reserve(complexVec.size());

                std::transform(
                    std::begin(complexVec),
                    std::end(complexVec),
                    std::back_inserter(realValues),
                    [](pmacc::math::Complex<T> const& data) { return data.get_real(); });
                return realValues;
            }

            template<typename T>
            std::vector<T> extractImag(std::vector<pmacc::math::Complex<T>> const& complexVec)
            {
                std::vector<T> imagValues;
                imagValues.reserve(complexVec.size());

                std::transform(
                    std::begin(complexVec),
                    std::end(complexVec),
                    std::back_inserter(imagValues),
                    [](pmacc::math::Complex<T> const& data) { return data.get_imag(); });
                return imagValues;
            }

            template<typename T>
            void copyVectorToBuffer(std::vector<T> const& vec, Buffer<T, DIM1>& buffer)
            {
                if(buffer.getCurrentSize() == vec.size())
                {
                    auto dataBox = buffer.getDataBox();
                    for(std::size_t ii = 0; ii < vec.size(); ii++)
                    {
                        dataBox[ii] = vec[ii];
                    }
                }
                else
                    throw std::runtime_error("XrayScattering: Tried to copy a vector"
                                             " to a Buffer of a different size");
            }
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
