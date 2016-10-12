/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
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

#include "compileTime/accessors/Identity.hpp"
#include "forward.hpp"

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/next_prior.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/bind.hpp>
#include <boost/type_traits.hpp>


/* Help to read this file:
 * - the definition of the operator() is outside of each class in a separate macro function
 * - the definition of the struct ForEach in namespace datail is a macro function *
 * - the macro definitions are inside of ####### TEXT #######
 */

#ifndef PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
/** number of parameters in operator()
 *
 * Maximal number of parameter which can be passed to ForEach `operator()` and
 * `operator() const`.
 * The count of parameters depends of the `operator()` from T_Functor.
 * If you need more than 10 parameter set  PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
 * to a higher number.
 */
#define PMACC_MAX_FUNCTOR_OPERATOR_PARAMS 10
#endif

namespace PMacc
{
namespace algorithms
{

/** macro creates  typeNameN & */
#define PMACC_UNUSED_REF(_, n, typeName) typeName##n &

//########################### definitions for preprocessor #####################
/** create operator() for EmptyFunctor
 *
 * template<typename T0, ... , typename TN>
 * HDINLINE void operator()(const T0&, ..., const TN& ) const {}
 *
 * All operator are empty and do nothing.
 * The operator parameters are plain type-only (without a name) to support
 * compiler flags like -Wextras (with -Wunused-params)
 */
#define PMACC_FOREACH_OPERATOR_NO_USAGE(Z, N, PMACC_PP_CONST)                  \
    PMACC_NO_NVCC_HDWARNING                                                    \
    /* if N != 0 we add `template<` */                                         \
    BOOST_PP_IF(BOOST_PP_NOT_EQUAL(N,0),template<,BOOST_PP_EMPTY())            \
    /*      <typename T0, ... , typename TN     > */                           \
    BOOST_PP_ENUM_PARAMS(N, typename T)                                        \
    /*  if N != 0 we add ```>``` */                                            \
    BOOST_PP_IF(BOOST_PP_NOT_EQUAL(N,0),>,BOOST_PP_EMPTY())                    \
    HDINLINE void                                                              \
    /*        ( const T0& , ... , const TN&                ) */                \
    operator()( BOOST_PP_ENUM(N, PMACC_UNUSED_REF, const T)) PMACC_PP_CONST    \
    {                                                                          \
    }/*end of operator()*/


/** makro creates getForwardedValue(tn) */
#define PMACC_GET_FORWARDED_VALUE(_, n, value_name) getForwardedValue( value_name##n )

/** create operator() for ForEach
 *
 * template<typename T0, ... , typename TN>
 * HDINLINE void operator()(const T0&, ..., const TN&) const {}
 */
#define PMACC_FOREACH_OPERATOR(Z, N, PMACC_PP_CONST)                           \
    PMACC_NO_NVCC_HDWARNING                                                    \
    /* if N != 0 we add `template<` */                                         \
    BOOST_PP_IF(BOOST_PP_NOT_EQUAL(N,0),template<,BOOST_PP_EMPTY())            \
    /* typename T0, ... , typename TN      */                                  \
    BOOST_PP_ENUM_PARAMS(N, typename T)                                        \
    /*  if N != 0 we add ```>``` */                                            \
    BOOST_PP_IF(BOOST_PP_NOT_EQUAL(N,0),>,BOOST_PP_EMPTY())                    \
    HDINLINE void                                                              \
    /*        ( const T0& t0, ... , const TN& tN           ) */                \
    operator()( BOOST_PP_ENUM_BINARY_PARAMS(N, const T, &t)) PMACC_PP_CONST    \
    {                                                                          \
        /*       (getForwardedValue(t0), ..., getForwardedValue(tn) ) */       \
        Functor()(BOOST_PP_ENUM(N,PMACC_GET_FORWARDED_VALUE, t));              \
        /*        (t0, ..., tn               ) */                              \
        NextCall()(BOOST_PP_ENUM_PARAMS(N, t));                                \
    } /*end of operator()*/

//########################### end preprocessor definitions #####################

namespace forEach
{
namespace detail
{
/** call the functor were itBegin points to
 *
 *  \tparam itBegin iterator to an element in a mpl sequence
 *  \tparam itEnd iterator to the end of a mpl sequence
 *  \tparam isEnd true if itBegin==itEnd, else false
 */
template<
typename itBegin,
typename itEnd,
bool isEnd = boost::is_same<itBegin, itEnd>::value >
struct CallFunctorOfIterator
{
    typedef typename boost::mpl::next<itBegin>::type nextIt;
    typedef typename boost::mpl::deref<itBegin>::type Functor;
    typedef CallFunctorOfIterator< nextIt, itEnd> NextCall;

    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * template<typename T0, ... , typename TN>
     * create operator()(const T0&, ..., const TN&) const {} */
    BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR, const)

    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * template<typename T0, ... , typename TN>
     * create operator()(const T0&, ..., const TN&) {} */
    BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR, BOOST_PP_EMPTY())
};

/** Recursion end of ForEach */
template<
typename itBegin,
typename itEnd>
struct CallFunctorOfIterator<itBegin, itEnd, true>
{
    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * create:
     * template<typename T0, ... , TN>
     * void operator()(const T0&, ..., const TN&) const {} */
    BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR_NO_USAGE, const)


    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * create:
     * template<typename T0, ... , TN>
     * void operator()(const T0&, ..., const TN&) {} */
    BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR_NO_USAGE, BOOST_PP_EMPTY())
};

} // namespace detail

/** Compile-Time for each for Boost::MPL Type Lists
 *
 *  \tparam T_MPLSeq A mpl sequence that can be accessed by mpl::begin, mpl::end, mpl::next
 *  \tparam T_Functor An unary lambda functor with a HDINLINE void operator()(...) method
 *          _1 is substituted by Accessor's result using boost::mpl::apply with elements from T_MPLSeq.
 *          The maximum number of parameters for the operator() is limited by
 *          PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
 *  \tparam T_Accessor An unary lambda operation
 *
 * Example:
 *      MPLSeq = boost::mpl::vector<int,float>
 *      Functor = any unary lambda functor
 *      Accessor = lambda operation identity
 *
 *      definition: F(X) means boost::apply<F,X>
 *
 *      call:   ForEach<MPLSeq,Functor,Accessor>()(42);
 *      unrolled code: Functor(Accessor(int))(42);
 *                     Functor(Accessor(float))(42);
 */

template<typename T_MPLSeq, typename T_Functor, class T_Accessor = compileTime::accessors::Identity<> >
struct ForEach
{

    template<typename X>
    struct ReplacePlaceholder : bmpl::apply1<T_Functor, typename bmpl::apply1<T_Accessor, X>::type >
    {
    };

    typedef typename bmpl::transform<
    T_MPLSeq,
    ReplacePlaceholder<bmpl::_1>
    >::type SolvedFunctors;

    typedef typename boost::mpl::begin<SolvedFunctors>::type begin;
    typedef typename boost::mpl::end< SolvedFunctors>::type end;


    typedef detail::CallFunctorOfIterator< begin, end > NextCall;
    /* this functor does nothing */
    typedef detail::CallFunctorOfIterator< end, end > Functor;

    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * template<typename T0, ... , typename TN>
     * create operator()(const T0&, ..., const TN&) const {}*/
    BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR, const)


    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * template<typename T0, ... , typename TN>
     * create operator()(const T0& t0,...,const TN& tN) {}*/
    BOOST_PP_REPEAT_FROM_TO(0, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR, BOOST_PP_EMPTY())
};

/* delete all preprocessor defines to avoid conflicts in other files */
#undef PMACC_FOREACH_OPERATOR
#undef PMACC_FOREACH_OPERATOR_NO_USAGE
#undef PMACC_GET_FORWARDED_VALUE

} // namespace forEach
} // namespace algorithms
} // namespace PMacc
