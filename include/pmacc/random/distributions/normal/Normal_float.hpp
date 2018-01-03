/* Copyright 2015-2018 Alexander Grund, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/random/distributions/Normal.hpp"
#include "pmacc/random/distributions/misc/MullerBox.hpp"
#include "pmacc/random/methods/XorMin.hpp"
#include "pmacc/random/methods/MRG32k3aMin.hpp"
#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/algorithms/math.hpp"


namespace pmacc
{
namespace random
{
namespace distributions
{
namespace detail
{

    //!Returns a random float value in [0,1) with normal distribution
    template< typename T_RNGMethod>
    class Normal<float, T_RNGMethod, void>
    {
        using RNGMethod = T_RNGMethod;
        using StateType = typename RNGMethod::StateType;
    public:
        using result_type = float;

        template< typename T_Acc >
        DINLINE result_type
        operator()(
            T_Acc const & acc,
            StateType& state
        )
        {
            return ::alpaka::rand::distribution::createNormalReal< float >(
                acc
            )( state );
        }
    };

    //! specialization for XorMin
    template<
        typename T_Acc
    >
    struct Normal<
        float,
        methods::XorMin< T_Acc >,
        void
    > :
        public MullerBox<
            float,
            methods::XorMin< T_Acc >
        >
    {

    };

    //! specialization for MRG32k3aMin
    template<
        typename T_Acc
    >
    struct Normal<
        float,
        methods::MRG32k3aMin< T_Acc >,
        void
    > :
        public MullerBox<
            float,
            methods::MRG32k3aMin< T_Acc >
        >
    {

    };
}  // namespace detail
}  // namespace distributions
}  // namespace random
}  // namespace pmacc
