/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include "types.h"

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>

namespace particleAccess
{

#define TEMPLATE_ARGS(Z, N, _) typename Arg ## N
#define NORMAL_ARGS(Z, N, _) Arg ## N arg ## N

#define CELL2PARTICLE_OPERATOR(Z, N, _) \
    template<typename TParticlesBox, typename CellIndex, typename Functor BOOST_PP_ENUM_TRAILING(N, TEMPLATE_ARGS, _)> \
    DINLINE void operator()(TParticlesBox pb, const CellIndex& cellIndex, Functor functor BOOST_PP_ENUM_TRAILING(N, NORMAL_ARGS, _)); \


//template<typename ParticleAttributes>
template<typename SuperCellSize>
struct Cell2Particle
{
    typedef void result_type;

    BOOST_PP_REPEAT(5, CELL2PARTICLE_OPERATOR, _)
};

#undef CELL2PARTICLE_OPERATOR
#undef TEMPLATE_ARGS
#undef NORMAL_ARGS

}

#include "Cell2Particle.tpp"

