/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
