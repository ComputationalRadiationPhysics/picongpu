/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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

/* create a random number generator on gpu
 * \tparam RngMethod method to generate random number
 * \tparam Distribution functor for distribution
 */
template<class RNGMethod, class Distribution>
class RNG : public RNGMethod
{
public:

    typedef RNG<RNGMethod, Distribution> This;

    HDINLINE RNG()
    {
    }

    /*
     * \param rngMethod instance of generator
     * \param distribution instance of distribution functor
     */
    DINLINE RNG(const RNGMethod& rng_method, const Distribution& rng_operation) :
    RNGMethod(rng_method), op(rng_operation)
    {
    }

    HDINLINE RNG(const This& other) :
    RNGMethod(static_cast<RNGMethod>(other)), op(other.op)
    {
    }

    /* default method to generate a random number
     * @return random number
     */
    DINLINE typename Distribution::Type operator()()
    {
        return this->op(this->getStatePtr());
    }

private:
    PMACC_ALIGN(op, Distribution);
};

/* create a random number generator on gpu
 * \tparam RngMethod method to generate random number
 * \tparam Distribution functor for distribution
 *
 * \param rngMethod instance of generator
 * \param distribution instance of distribution functor
 * \return class which can used to generate random numbers
 */
template<class RngMethod, class Distribution>
DINLINE typename PMacc::nvidia::rng::RNG<RngMethod, Distribution> create(const RngMethod & rngMethod,
                                                                         const Distribution & distribution)
{
    return PMacc::nvidia::rng::RNG<RngMethod, Distribution > (rngMethod, distribution);
}

}
}
}
