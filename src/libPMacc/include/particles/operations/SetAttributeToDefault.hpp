/**
 * Copyright 2014 Rene Widera
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

#include "types.h"
#include "traits/Resolve.hpp"

namespace PMacc
{

/** set an attribute of a particle to its default value
 *
 * @tparam  T_Attribute value_identifier or alias which is a value_identifier
 */
template<typename T_Attribute>
struct SetAttributeToDefault
{
    typedef T_Attribute Attribute;

    /** set an attribute to their default value
     *
     * @tparam T_Partcile particle type
     */
    template<typename T_Particle>
    HDINLINE
    void operator()(T_Particle& particle)
    {
        typedef typename PMacc::traits::Resolve<Attribute>::type ResolvedAttr;
        /* set attribute to it's user defined default value */
        particle[Attribute()] = ResolvedAttr::getValue();
    }
};


}//namespace PMacc
