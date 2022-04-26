/* Copyright 2022 Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/meta/conversion/ToSeq.hpp"

#include <boost/mpl/fold.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>


namespace pmacc
{
    /** Make a sequence out of the input sequence with the duplicate elements removed
     *
     * The new sequence may have the elements in different order.
     *
     * @tparam T_Seq source sequence
     */
    template<typename T_Seq>
    struct Unique
    {
        // Insert all sequence elements to a set, that will remove duplicates.
        // Note that we cannot simply call bmpl::unique as it only removes duplicates located contiguously.
        using Set = typename bmpl::fold<T_Seq, bmpl::set0<>, bmpl::insert<bmpl::_1, bmpl::_2>>::type;

        // Convert back to sequence if necessary
        using type = typename ToSeq<Set>::type;
    };

    //! Helper alias for @see Unique<>
    template<typename T_Seq>
    using Unique_t = typename Unique<T_Seq>::type;

    /** Flag whether the given sequence is unique (has no duplicate elements)
     *
     * @tparam T_Seq sequence
     */
    template<typename T_Seq>
    static constexpr bool isUnique = (bmpl::size<T_Seq>::value == bmpl::size<Unique_t<T_Seq>>::value);

} // namespace pmacc
