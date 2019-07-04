/* Copyright 2019 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/fields/manipulator/Unary.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/fields/algorithm/CallForEach.hpp>
#include <pmacc/meta/conversion/ToSeq.hpp>
#include <pmacc/meta/ForEach.hpp>

#include <boost/mpl/placeholders.hpp>


namespace picongpu
{
namespace fields
{
namespace detail
{

    /** Operator to create a functor
     */
    template<
        typename T_Manipulator,
        typename T_Field
    >
    struct MakeFunctorOperator
    {
    public:
        using type = bmpl::apply1<
            T_Manipulator,
            T_Field
        >;
    };

} // namespace detail

template<
    typename T_Manipulator,
    typename T_Field
>
struct Helper{
    using type = typename bmpl::apply1<
        T_Manipulator,
        T_Field
    >;//::type;
//using type = int;
};

    /// TODO
    /** Run a manipulator for each value of the given field
     *
     * Allows to manipulate attributes of existing particles in a species with
     * arbitrary unary functors ("manipulators").
     *
     * @tparam T_Manipulator unary lambda functor accepting one particle
     *                       species,
     *                       @see picongpu::particles::manipulators
     * @tparam T_Field field type
     */
    template<
        typename T_Manipulator,
        typename T_Field = bmpl::_1
    >
    using Manipulate = pmacc::fields::algorithm::CallForEach<
        T_Field,
        //bmpl::apply1<
        //    T_Manipulator,
        //    T_Field
        //>
        detail::MakeFunctorOperator< T_Manipulator, T_Field> //Helper< T_Manipulator, T_Field>
    >;
    //struct Manipulate
    //{

    //    void operator()(uint32_t currentStep)
    //    {
    //        using Operator = detail::MakeFunctorOperator<
    //            T_Manipulator,
    //            T_Field
    //        >;
    //        Operator o;
    //        using OperatorType = Operator::type;
    //        OperatorType ot( currentStep );
    //        pmacc::fields::algorithm::CallForEach< T_Field, Operator > callForEach;
    //        callForEach( currentStep );
    //    }
    //};

    template<
        typename T_Manipulator,
        typename T_Fields
    >
    HINLINE void manipulate( uint32_t const currentStep )
    {
        using FieldsSeq = typename pmacc::ToSeq< T_Fields >::type;
        using Functor = Manipulate<
            T_Manipulator,
            bmpl::_1
        >;
        pmacc::meta::ForEach<
            FieldsSeq,
            Functor
        > forEach;
        forEach( currentStep );
    }

} // namespace fields
} // namespace picongpu
