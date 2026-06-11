/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/SuperCellDescription.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /** execute a functor for each cell of a domain
     *
     * the user functor is executed on each elements of the full domain (GUARD +CORE)
     *
     * @tparam T_DataDomain pmacc::SuperCellDescription, compile time data domain
     *                      description with a CORE and GUARD
     */
    template<typename T_DataDomain>
    class ThreadCollective
    {
    private:
        // size of the CORE (in elements per dimension)
        using CoreDomainSize = typename T_DataDomain::SuperCellSize;
        // full size of the domain including the GUARD (in elements per dimension)
        using DomainSize = typename T_DataDomain::FullSuperCellSize;
        // offset (in elements per dimension) from the GUARD origin to the CORE
        using OffsetOrigin = typename T_DataDomain::OffsetOrigin;

        static constexpr uint32_t dim = T_DataDomain::Dim;

    public:
        /** execute the user functor for each element in the full domain
         *
         * @tparam T_Worker lockstep worker type
         * @tparam T_Functor type of the user functor, must have a `void operator()`
         *                   with as many arguments as args contains
         * @tparam T_Args type of the arguments, each type must implement an operator
         *                 `template<typename T, typename R> R operator(T)`
         *
         * @param worker lockstep worker
         * @param functor user defined functor
         * @param args arguments passed to the functor
         *             The method `template<typename T, typename R> R operator(T)`
         *             is called for each argument, the result is passed to the
         *             functor `functor::operator()`.
         *             `T` is a N-dimensional vector of an index relative to the origin
         *             of data domain GUARD
         */
        template<typename T_Worker, typename T_Functor, typename... T_Args>
        DINLINE void operator()(T_Worker const& worker, T_Functor& functor, T_Args&&... args)
        {
            lockstep::makeForEach<math::CT::volume<DomainSize>::type::value>(worker)(
                [&](int32_t const linearIdx)
                {
                    /* offset (in elements) of the current processed element relative
                     * to the origin of the core domain
                     */
                    DataSpace<dim> const offset
                        = pmacc::math::mapToND(DomainSize::toRT(), linearIdx) - OffsetOrigin::toRT();
                    functor(worker, args(offset)...);
                });
        }
    };

    template<typename T_DataDomain>
    HDINLINE auto makeThreadCollective()
    {
        return ThreadCollective<T_DataDomain>{};
    }

    template<typename T_DataDomain>
    HDINLINE auto makeThreadCollective(T_DataDomain const& /*dataDomainSize*/)
    {
        return ThreadCollective<T_DataDomain>{};
    }

} // namespace pmacc
