/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "version.hpp"

#include <pmacc/algorithms/PromoteType.hpp>
#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/algorithms/math.hpp>
#include <pmacc/math/math.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>

namespace picongpu
{
    namespace precision32Bit
    {
        using precisionType = float;
    } // namespace precision32Bit

    namespace precision64Bit
    {
        using precisionType = double;
    } // namespace precision64Bit

    namespace math = pmacc::math;
    /** g++ 9 creates compile issues when pulling definitions into picongpu namepsace via 'using namespace
     * pmacc::algorithms::precisionCast;' therefore we pull the class and function separate
     */
    using pmacc::algorithms::precisionCast::precisionCast;
    template<typename CastToType, typename Type>
    using TypeCast = pmacc::algorithms::precisionCast::TypeCast<CastToType, Type>;

    using namespace pmacc::algorithms::promoteType;
    using namespace pmacc::traits;

} // namespace picongpu
