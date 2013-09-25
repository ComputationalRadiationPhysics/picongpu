/**
 * Copyright 2013 René Widera
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
#include "particles/factories/GetKeyFromAlias.hpp"
#include "particles/factories/CopyIdentifier.hpp"
#include "algorithms/ForEach.hpp"
#include "RefWrapper.hpp"

#include "particles/operations/Assign.hpp"
#include "particles/operations/Deselect.hpp"
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/is_sequence.hpp>

namespace PMacc
{
namespace bmpl = boost::mpl;
namespace pmath = PMacc::math;
namespace pmacc = PMacc;

template<typename T_FrameType, typename T_ValueTypeSeq = typename T_FrameType::ValueTypeSeq>
struct Particle : public InheritLinearly<typename T_FrameType::MethodsList>
{
    typedef T_FrameType FrameType;
    typedef T_ValueTypeSeq ValueTypeSeq;
    typedef Particle<FrameType, ValueTypeSeq> ThisType;
    typedef typename FrameType::MethodsList MethodsList;


    FrameType& frame;
    uint32_t idx;

    HDINLINE Particle(FrameType& frame, uint32_t idx) : frame(frame), idx(idx)
    {
    }

    template<typename T_OtherParticle >
    HDINLINE Particle(const T_OtherParticle& other) : frame(other.frame), idx(other.idx)
    {
    }

    template<typename T_Key >
    HDINLINE
    typename boost::result_of<
    typename boost::remove_reference<
    typename boost::result_of < FrameType(typename GetKeyFromAlias_assert<ValueTypeSeq, T_Key>::type)>::type
    >::type(uint32_t)
    >::type
    operator[](const T_Key key)
    {
        return frame.getIdentifier(key)[idx];
    }

    template<typename T_Key >
    HDINLINE
    typename boost::result_of<
    typename boost::remove_reference<
    typename boost::result_of < FrameType(typename GetKeyFromAlias_assert<ValueTypeSeq, T_Key>::type)>::type
    >::type(uint32_t)
    >::type
    operator[](const T_Key key) const
    {
        return frame.getIdentifier(key)[idx];
    }
    /*
        template<typename T_OtherParticle >
            HDINLINE
            ThisType& operator=(const T_OtherParticle& other)
        {
            particles::operations::assign(*this, other);
            return *this;
        }

        HDINLINE
        ThisType& operator=(const ThisType& other)
        {
            particles::operations::assign(*this, other);
            return *this;
        }
     */
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
    void operator()(const Dest& dest, const Src& src)
    {
        algorithms::forEach::ForEach<typename Dest::ValueTypeSeq,
            CopyIdentifier<void> > copy;
        copy(byRef(dest), src);
    };
};

template<
typename T_RemoveSequence,
typename T_FrameType, typename T_ValueTypeSeq
>
struct Deselect
<
T_RemoveSequence,
PMacc::Particle<T_FrameType, T_ValueTypeSeq>
>
{
    typedef T_FrameType FrameType;
    typedef T_ValueTypeSeq ValueTypeSeq;
    typedef PMacc::Particle<FrameType, ValueTypeSeq> Object;


    typedef T_RemoveSequence RemoveSequence;

    BOOST_MPL_ASSERT((boost::mpl::is_sequence< RemoveSequence >));

    template<typename T_Key>
    struct hasId
    {
        typedef typename GetKeyFromAlias<ValueTypeSeq, T_Key>::type Key;
        typedef bmpl::bool_<!boost::is_same< bmpl::void_, Key>::value> type;
    };

    typedef typename bmpl::remove_if< ValueTypeSeq, hasId<bmpl::_> >::type NewValueTypeSeq;

    typedef PMacc::Particle<FrameType, NewValueTypeSeq> result;

    HDINLINE
    result operator()(const Object& particle)
    {
        return result(particle);
    };
};


} //namespace detail
} //namespace operations
} //namespace particles

} //namespace PMacc
