/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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



#ifndef GAMMA_HPP
#define    GAMMA_HPP

namespace picongpu
{

using namespace PMacc;

template<typename precisionType = float_X>
struct Gamma
{
    typedef precisionType valueType;

    template<typename MomType, typename MassType >
        HDINLINE valueType operator()(const MomType mom, const MassType mass)
    {
        const valueType fMom2 = math::abs2( precisionCast<valueType >( mom ) );
        const valueType c2 = SPEED_OF_LIGHT*SPEED_OF_LIGHT;

        const valueType m2_c2_reci = valueType(1.0) / precisionCast<valueType >( mass * mass * c2 );

        return math::sqrt( precisionCast<valueType >( valueType(1.0) + fMom2 * m2_c2_reci ) );
    }


};

}

#endif    /* GAMMA_HPP */

