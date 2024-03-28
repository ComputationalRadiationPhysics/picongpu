/* Copyright 2022-2023 Sergei Bastrakov
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

#include "pmacc/meta/conversion/MakeSeq.hpp"
#include "pmacc/meta/conversion/ToSeq.hpp"

#include <boost/mpl/empty.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/remove.hpp>
#include <boost/mpl/size.hpp>


namespace pmacc
{
    /** Make a sequence out of the input sequence with the duplicates of elements removed
     *
     * Each element present in the input sequence is present in the result, and present exactly once.
     * This operation turned out surprisingly tricky to implement with boost::mpl, see #4078 for details.
     * We implement it in a simplistic and inefficient manner regarding compile time.
     * However here it is not an issue as this is not a core metaprogramming routine.
     *
     * @tparam T_Seq source sequence
     * @tparam T_isEmpty whether the source sequence is empty
     *
     * @{
     */

    /** General implementation for non-empty sequences
     *
     * Take the front element, remove its other instances from the rest of the sequence,
     * recursively repeat for the remaining elements.
     */
    template<typename T_Seq, bool T_isEmpty = mp_empty<T_Seq>::value>
    struct Unique
    {
        using Front = mp_front<T_Seq>;
        using Tail = mp_remove<T_Seq, Front>;
        using UniqueTail = typename Unique<Tail>::type;
        using type = MakeSeq_t<Front, UniqueTail>;
    };

    //! Specialization for empty sequences
    template<typename T_Seq>
    struct Unique<T_Seq, true>
    {
        using type = MakeSeq_t<>;
    };

    /** }@ */

    //! Helper alias for @see Unique<>
    template<typename T_Seq>
    using Unique_t = typename Unique<T_Seq>::type;

    /** Flag whether the given sequence is unique (has no duplicate elements)
     *
     * @tparam T_Seq sequence
     */
    template<typename T_Seq>
    static constexpr bool isUnique = (mp_size<T_Seq>::value == mp_size<Unique_t<T_Seq>>::value);

} // namespace pmacc
