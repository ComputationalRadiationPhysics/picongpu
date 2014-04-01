/**
 * Copyright 2013 Rene Widera
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

#include "types.h"
#include "particles/boostExtension/InheritLinearly.hpp"
#include <boost/utility/result_of.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>
#include "traits/HasIdentifier.hpp"
#include "compileTime/GetKeyFromAlias.hpp"
#include "compileTime/conversion/ResolveAliases.hpp"
#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "particles/operations/CopyIdentifier.hpp"
#include "algorithms/ForEach.hpp"
#include "RefWrapper.hpp"

#include "particles/operations/Assign.hpp"
#include "particles/operations/Deselect.hpp"
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/contains.hpp>

namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;

/** A single particle of a @see Frame
 * 
 * A instance of this Particle is only a reference to a dataset of @see Frame
 * 
 * @tparam T_FrameType type of the parent frame
 * @tparam T_ValueTypeSeq sequence with all attribute identifier 
 *                        (can be a subset of T_FrameType::ValueTypeSeq)
 */
template<typename T_FrameType, typename T_ValueTypeSeq = typename T_FrameType::ValueTypeSeq>
struct Particle : public InheritLinearly<typename T_FrameType::MethodsList>
{
    typedef T_FrameType FrameType;
    typedef T_ValueTypeSeq ValueTypeSeq;
    typedef typename FrameType::Name Name;
    typedef Particle<FrameType, ValueTypeSeq> ThisType;
    typedef typename FrameType::MethodsList MethodsList;

    /* IMPORTANT: store first value with big size to avoid
     * that pointer is copyed byte by byte because data are not alligned
     * in this case
     * 
     * in this case sizeof(uint32_t)>sizeof(reference)
     */
    /** index of particle inside the Frame*/
    const uint32_t idx;
    /** reference to parent frame where this particle is from*/
    FrameType& frame;
    
    /** create particle
     *
     * @param frame reference to parent frame
     * @param idx index of particle inside the frame
     */
    HDINLINE Particle(FrameType& frame, uint32_t idx) : frame(frame), idx(idx)
    {
    }

    template<typename T_OtherParticle >
    HDINLINE Particle(const T_OtherParticle& other) : frame(other.frame), idx(other.idx)
    {
    }

    /** access attribute with a identifier
     *
     * @param T_Key instance of identifier type 
     *              (can be an alias, value_identifier or any other class)
     * @return result of operator[] of the Frame
     */
    template<typename T_Key >
    HDINLINE
    typename boost::result_of<
    typename boost::remove_reference<
    typename boost::result_of < FrameType(T_Key)>::type
    >::type(uint32_t)
    >::type
    operator[](const T_Key key)
    {
        return frame.getIdentifier(key)[idx];
    }

    /** const version of method operator(const T_Key) */
    template<typename T_Key >
    HDINLINE
    typename boost::result_of<
    typename boost::remove_reference<
    typename boost::result_of <const FrameType(T_Key)>::type
    >::type(uint32_t)
    >::type
    operator[](const T_Key key) const
    {

        return frame.getIdentifier(key)[idx];
    }
private:
    /* we disallow to assign this class*/
    template<typename T_OtherParticle >
    HDINLINE
    ThisType& operator=(const T_OtherParticle& other);

    HDINLINE
    ThisType& operator=(const ThisType& other);
};

namespace traits
{

template<typename T_Key,
typename T_FrameType
>
struct HasIdentifier<
PMacc::Particle<T_FrameType>,
T_Key
>
{
private:
    typedef T_FrameType FrameType;
public:
    typedef typename HasIdentifier<FrameType, T_Key>::type type;
    static const bool value = type::value;
};
} //namespace traits

namespace particles
{
namespace operations
{
namespace detail
{

template<
typename T_FrameType1, typename T_ValueTypeSeq1,
typename T_FrameType2, typename T_ValueTypeSeq2
>
struct Assign
<
PMacc::Particle<T_FrameType1, T_ValueTypeSeq1>,
PMacc::Particle<T_FrameType2, T_ValueTypeSeq2>
>
{
    typedef PMacc::Particle<T_FrameType1, T_ValueTypeSeq1> Dest;
    typedef PMacc::Particle<T_FrameType2, T_ValueTypeSeq2> Src;

    HDINLINE
    void operator()(Dest& dest, const Src& src)
    {
        algorithms::forEach::ForEach<typename Dest::ValueTypeSeq,
            CopyIdentifier<void> > copy;
        copy(byRef(dest), src);
    };
};

template<
typename T_MPLSeqWithObjectsToRemove,
typename T_FrameType, typename T_ValueTypeSeq
>
struct Deselect
<
T_MPLSeqWithObjectsToRemove,
PMacc::Particle<T_FrameType, T_ValueTypeSeq>
>
{
    typedef T_FrameType FrameType;
    typedef T_ValueTypeSeq ValueTypeSeq;
    typedef PMacc::Particle<FrameType, ValueTypeSeq> ParticleType;
    typedef T_MPLSeqWithObjectsToRemove MPLSeqWithObjectsToRemove;
    
    /* translate aliases to full specialized identifier*/
    typedef typename ResolveAliases<MPLSeqWithObjectsToRemove, ValueTypeSeq>::type ResolvedSeqWithObjectsToRemove;
    /* remove types from original particle attribute list*/
    typedef typename RemoveFromSeq<ValueTypeSeq, ResolvedSeqWithObjectsToRemove>::type NewValueTypeSeq;
    /* new particle type*/
    typedef PMacc::Particle<FrameType, NewValueTypeSeq> ResultType;

    template<class> struct result;

    template<class F, class T_Obj>
    struct result< F(T_Obj)>
    {
        typedef ResultType type;
    };

    HDINLINE
    ResultType operator()(const ParticleType& particle)
    {
        return ResultType(particle);
    };
};

} //namespace detail
} //namespace operations
} //namespace particles

} //namespace PMacc
