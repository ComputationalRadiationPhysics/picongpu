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

                /*create a random float number from [0.0,1.0)
                 */
                class Uniform_float
                {
                public:
                    typedef float Type;

                    HDINLINE Uniform_float()
                    {
                    }

                    template<class RNGState>
                    DINLINE Type operator()(RNGState* state) const
                    {
                        // (0.f, 1.0f]
                        const Type raw = curand_uniform(state);

                        /// \warn hack, are is that really ok? I say, yes, since
                        /// it shifts just exactly one number. Axel
                        ///
                        ///   Note: (1.0f - raw) does not work, since
                        ///         nvidia seems to return denormalized
                        ///         floats around 0.f (thats not as they
                        ///         state it out in their documentation)
                        // [0.f, 1.0f)
                        const Type r = raw * static_cast<float>( raw != Type(1.0) );
                        return r;
                    }

                };
            }
        }
    }
}
