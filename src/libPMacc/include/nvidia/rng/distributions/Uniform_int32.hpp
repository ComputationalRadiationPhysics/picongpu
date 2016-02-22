/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include <curand_kernel.h>
#include "pmacc_types.hpp"

namespace PMacc
{
    namespace nvidia
    {
        namespace rng
        {
            namespace distributions
            {

                /*create a 32Bit random int number
                 * Range: [INT_MIN,INT_MAX]
                 */
                class Uniform_int32
                {
                public:
                    typedef int32_t Type;

                    HDINLINE Uniform_int()
                    {
                    }

                    template<class RNGState>
                    DINLINE Type operator()(RNGState* state) const
                    {
                        /*curand create a random 32Bit int value*/
                        return curand(state);
                    }
                };
            }
        }
    }

}
