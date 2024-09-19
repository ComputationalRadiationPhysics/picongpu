/* Copyright 2013-2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/traits/Resolve.hpp"

namespace pmacc
{
    /** copy an attribute of a particle from another particle
     *
     * @tparam T_Attribute value_identifier or alias which is a value_identifier
     *                     Attribute must be available in source and destination particle.
     */
    template<typename T_Attribute>
    struct CopyValueIdentifier
    {
        /** derive value from source particle and assign it to the destination */
        template<typename T_DestParticleType, typename T_SrcParticleType>
        HDINLINE void operator()(T_DestParticleType& destParticle, T_SrcParticleType const& srcParticle) const
        {
            using ResolvedAttr = typename pmacc::traits::Resolve<T_Attribute>::type;
            /* set attribute to its user defined default value */
            destParticle[T_Attribute()] = ResolvedAttr{}.copyValue(T_Attribute{}, srcParticle);
        }
    };

} // namespace pmacc
