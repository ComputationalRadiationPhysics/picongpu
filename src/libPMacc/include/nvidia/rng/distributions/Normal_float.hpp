/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
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


#ifndef NORMAL_FLOAT_HPP
#define	NORMAL_FLOAT_HPP

#include <curand_kernel.h>
#include "types.h"

namespace PMacc
{
    namespace nvidia
    {
        namespace rng
        {
            namespace distributions
            {

                /*Return normally distributed floats with mean 0.0f and standard deviation 1.0f
                 */
                class Normal_float
                {
                public:
                    typedef float Type;

                    DINLINE Normal_float()
                    {
                    }

                    template<class RNGState>
                    DINLINE Type operator()(RNGState* state) const
                    {
                        return curand_normal(state);
                    }

                };
            }
        }
    }
}


#endif	/* NORMAL_FLOAT_HPP */

