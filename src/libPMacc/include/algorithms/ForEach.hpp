/**
 * Copyright 2013 Axel Huebl, Heiko Burau, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/next_prior.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/bind.hpp>
#include <boost/type_traits.hpp>

#include <boost/mpl/if.hpp>

#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>


/* Help to read this file:
 * - the definition of the operator() is outside of each class in a separate macro function
 * - the definition of the struct ForEach in namespace datail is a macro function *
 * - the macro definitions are inside of ####### TEXT #######
 */

#ifndef PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
#define PMACC_MAX_FUNCTOR_OPERATOR_PARAMS 6
#endif

#ifndef PMACC_MAX_FUNCTOR_TEMPLATES
#define PMACC_MAX_FUNCTOR_TEMPLATES 4
#endif

namespace PMacc
{
namespace algorithms
{

//########################### definitions for preprocessor #####################
/** create operator() for EmptyFunctor
 * 
 * template<typename T0, ... , typename TN>
 * HDINLINE void operator()(const T0, ..., const TN ) const {}
 * 
 * All operator are empty and do nothing.
 * The operator parameters are plain type-only (without a name) to support 
 * compiler flags like -Wextras (with -Wunused-params)
 */
#define PMACC_FOREACH_OPERATOR_CONST_NO_USAGE(Z, N, FUNCTIONS)                 \
    /*      <typename T0, ... , typename TN     > */                           \
    PMACC_NO_NVCC_HDWARNING                                                    \
    template<BOOST_PP_ENUM_PARAMS(N, typename T)>                              \
    /*                      ( const T0, ... , cont TN         ) */             \
    HDINLINE void operator()( BOOST_PP_ENUM_PARAMS(N, const T)) const          \
    {                                                                          \
    }/*end of operator()*/
//########################### end preprocessor definitions #####################

/** Empty functor class with operator() with N parameters
 */
struct EmptyFunctor
{
    PMACC_NO_NVCC_HDWARNING
    HDINLINE void operator()() const
    {
    }

    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * create:
     * template<typename T0, ... , TN> 
     * void operator()(const T0 ,...,const TN){}
     */
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR_CONST_NO_USAGE, _)
};

/* delete all preprocessor defines to avoid conflicts in other files */
#undef PMACC_FOREACH_OPERATOR_CONST_NO_USAGE

namespace forEach
{
namespace detail
{

template<typename Accessor, typename itBegin, typename itEnd, typename Functor>
struct ForEach;

//########################### definition for preprocessor ######################
/** create operator() for ForEach
 * 
 * template<typename T0, ... , typename TN>
 * HDINLINE void operator()(const T0 t0, ..., const TN tN) const {}
 */
#define PMACC_FOREACH_OPERATOR_CONST(Z, N, _)                                  \
    PMACC_NO_NVCC_HDWARNING                                                    \
    /*      <typename T0, ... , typename TN     > */                           \
    template<BOOST_PP_ENUM_PARAMS(N, typename T)>                              \
    /*                      ( const T0& t0, ... , const TN& tN           ) */  \
    HDINLINE void operator()( BOOST_PP_ENUM_BINARY_PARAMS(N, const T, &t)) const \
    {                                                                          \
        /*           (t0, ..., tn               ) */                           \
        FunctorType()(BOOST_PP_ENUM_PARAMS(N, t));                             \
        /*        (t0, ..., tn               ) */                              \
        NextCall()(BOOST_PP_ENUM_PARAMS(N, t));                                \
    } /*end of operator()*/

/** write a comma (,)*/
#define COMMA ,
/** write word in text*/
#define TEXT(Z, N, text)   text 
/** write comma then word in text and append N to text*/
#define TEXTAPPEND(Z, N, text)  COMMA  text ## N


/** Create:
 *  struct ForEach<Accessor,iteratorBegin,iteratorEnd,template<typename,...> 
 *  class Functor>{}
 * 
 *  \tparam Accessor A class with one template argument where ::type is defined.
 *          Accessor<TypeFromSeq>::type is called -> return type is given to Functor
 *          as T0.
 *  \tparam itBegin iterator to an element in a mpl sequence
 *  \tparam itEnd iterator to the end of a mpl sequence
 *  \tparam Functor A functor<T0,...,TN> with a HDINLINE void operator()(...) method
 *          T0 can be any type and is substituted by types from MPLSeq.
 *          The maximum number of template parameters of the functor Functor is 
 *          limited by PMACC_MAX_FUNCTOR_TEMPLATES.
 *          The maximum number of parameters for the operator() is limited by 
 *          PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
 */
#define PMACC_FOREACH_STRUCT(Z,N,_)                                            \
template<                                                                      \
    template<typename> class Accessor,                                         \
    typename AccessorType,                                                     \
    typename itBegin,                                                          \
    typename itEnd,                                                            \
    /*      <typename , ... , typename       > */                              \
    template<BOOST_PP_ENUM(N, TEXT, typename)> class Functor_,                 \
    /* typename A0, ... , typename AN  */                                      \
    BOOST_PP_ENUM_PARAMS(N, typename A)>                                       \
    /*                                                          <A0, ... , AN              > */ \
struct ForEach< Accessor<AccessorType>, itBegin, itEnd, Functor_<BOOST_PP_ENUM_PARAMS(N, A)> >  \
{                                                                              \
    typedef typename boost::mpl::next<itBegin>::type nextIt;                   \
    typedef typename boost::mpl::deref<itBegin>::type usedType;                \
    typedef typename boost::is_same<nextIt, itEnd>::type isEnd;                \
    /*              <A0, ... , AN              > */                            \
    typedef Functor_<BOOST_PP_ENUM_PARAMS(N, A)> FunctorUnchanged;             \
    typedef Functor_<typename Accessor<usedType>::type                         \
    /* Nothing is done if N equal 1                                            \     
            ,A1, ... , AN */                                                   \
            BOOST_PP_REPEAT_FROM_TO(1,N,TEXTAPPEND, A) > FunctorType;          \
    typedef detail::ForEach< Accessor<AccessorType>, nextIt, itEnd, FunctorUnchanged > TmpNextCall; \
    /* if nextIt is equal to itEnd we use EmptyFunctor                         \
       and end recursive call of EachFunctor */                                \
    typedef typename boost::mpl::if_< isEnd,                                   \
                                      EmptyFunctor,                            \
                                      TmpNextCall>::type NextCall;             \
                                                                               \
    PMACC_NO_NVCC_HDWARNING                                                    \
    HDINLINE void operator()() const                                           \
    {                                                                          \
        FunctorType()();                                                       \
        NextCall()();                                                          \
    }                                                                          \
    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS                                     \
     * template<typename T0, ... , typename TN>                                \
     * create operator()(const T0 t0,...,const TN tN)                          \
     */                                                                        \
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS), \
                            PMACC_FOREACH_OPERATOR_CONST, _)                   \
}; /*end of struct ForEach*/

//########################### end preprocessor definitions #####################

/* N = PMACC_MAX_FUNCTOR_TEMPLATES
 * create: 
 * struct definitions ForEach<Accessor,itBegin,itEnd,T0,...,TN>{}
 *  \see PMACC_FOREACH_STRUCT
 */
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(PMACC_MAX_FUNCTOR_TEMPLATES), PMACC_FOREACH_STRUCT, _)


/* delete all preprocessor defines to avoid conflicts in other files */
#undef PMACC_FOREACH_STRUCT
#undef COMMA
#undef TEXT
#undef TEXTAPPEND

} // namespace detail

/** Compile-Time for each for Boost::MPL Type Lists
 * 
 *  \tparam MPLSeq A mpl sequence that can be accessed by mpl::begin, mpl::end, mpl::next
 *  \tparam Functor A functor<T0,...,TN> with a HDINLINE void operator()(...) method
 *          T0 can be any type and is substituted by types from MPLSeq.
 *          The maximum number of template parameters of the functor Functor is 
 *          limited by PMACC_MAX_FUNCTOR_TEMPLATES.
 *          The maximum number of parameters for the operator() is limited by 
 *          PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
 *  \tparam Accessor A class with one template argument where ::type is defined.
 *          Accessor<TypeFromSeq>::type is called -> return type is given to Functor
 *          as T0.
 *  
 */
template<typename MPLSeq, typename Functor, class Accessor = compileTime::accessors::Identity<void> >
struct ForEach
{
    typedef typename boost::mpl::begin<MPLSeq>::type begin;
    typedef typename boost::mpl::end< MPLSeq>::type end;

    typedef typename boost::is_same<begin, end>::type isEnd;
    typedef detail::ForEach<Accessor, begin, end, Functor > TmpNextCall;
    /* if MPLSeq is empty we use EmptyFunctor */
    typedef typename boost::mpl::if_<isEnd, EmptyFunctor, TmpNextCall>::type NextCall;
    typedef EmptyFunctor FunctorType;

    PMACC_NO_NVCC_HDWARNING
    HDINLINE void operator()() const
    {
        FunctorType()();
        NextCall()();
    }

    /* N=PMACC_MAX_FUNCTOR_OPERATOR_PARAMS
     * template<typename T0, ... , typename TN>
     * create operator()(const T0 t0,...,const TN tN)
     */
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(PMACC_MAX_FUNCTOR_OPERATOR_PARAMS),
                            PMACC_FOREACH_OPERATOR_CONST, _)
};

/* delete all preprocessor defines to avoid conflicts in other files */
#undef PMACC_FOREACH_OPERATOR_CONST

} // namespace forEach
} // namespace algorithms
} // namespace PMacc
