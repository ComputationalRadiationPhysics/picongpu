/* Copyright 2024 Rene Widera
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

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/kernel/atomic.hpp"
#include "pmacc/type/Integral.hpp"

namespace pmacc::idDetail
{
    ALPAKA_FN_ACC uint64_t nextId;

    ALPAKA_FN_ACC uint64_t fetchAddId()
    {
        return kernel::atomicAllInc(&nextId);
    }

    ALPAKA_FN_ACC uint64_t& getIdRef()
    {
        return nextId;
    }
} // namespace pmacc::idDetail
