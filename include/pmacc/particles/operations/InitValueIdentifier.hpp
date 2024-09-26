/* Copyright 2014-2023 Rene Widera
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

#include "pmacc/identifier/value_identifier.hpp"
#include "pmacc/traits/Resolve.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** set an attribute of a particle to its default value
     *
     * @tparam  T_Attribute value_identifier or alias which is a value_identifier
     */
    template<typename T_Attribute>
    struct InitValueIdentifier
    {
        using Attribute = T_Attribute;

        template<typename T_Worker, typename T_DestParticleType>
        HDINLINE void operator()(T_Worker const& worker, IdGenerator idGen, T_DestParticleType& destParticle) const
        {
            using ResolvedAttr = typename pmacc::traits::Resolve<Attribute>::type;
            /* set attribute to its user defined default value */
            destParticle[Attribute{}] = ResolvedAttr{}.initValue(worker, idGen);
        }
    };


} // namespace pmacc
