/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
