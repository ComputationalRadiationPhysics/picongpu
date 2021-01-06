/* Copyright 2013-2021 Rene Widera
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

#include "pmacc/types.hpp"
#include "pmacc/particles/boostExtension/InheritLinearly.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/meta/GetKeyFromAlias.hpp"
#include "pmacc/meta/conversion/ResolveAliases.hpp"
#include "pmacc/meta/conversion/RemoveFromSeq.hpp"
#include "pmacc/particles/operations/CopyIdentifier.hpp"
#include "pmacc/meta/ForEach.hpp"
#include "pmacc/static_assert.hpp"

#include "pmacc/particles/operations/Assign.hpp"
#include "pmacc/particles/operations/Deselect.hpp"
#include "pmacc/particles/operations/SetAttributeToDefault.hpp"
#include "pmacc/meta/errorHandlerPolicies/ReturnValue.hpp"
#include <boost/utility/result_of.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/remove_if.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/not.hpp>

namespace pmacc
{
    namespace pmath = pmacc::math;

    /** A single particle of a @see Frame
     *
     * A instance of this Particle is a representation ("pointer") to the memory
     * where the frame is stored.
     *
     * @tparam T_FrameType type of the parent frame
     * @tparam T_ValueTypeSeq sequence with all attribute identifiers
     *                        (can be a subset of T_FrameType::ValueTypeSeq)
     */
    template<typename T_FrameType, typename T_ValueTypeSeq = typename T_FrameType::ValueTypeSeq>
    struct Particle : public InheritLinearly<typename T_FrameType::MethodsList>
    {
        typedef T_FrameType FrameType;
        typedef T_ValueTypeSeq ValueTypeSeq;
        typedef typename FrameType::Name Name;
        typedef typename FrameType::SuperCellSize SuperCellSize;
        typedef Particle<FrameType, ValueTypeSeq> ThisType;
        typedef typename FrameType::MethodsList MethodsList;

        /** index of particle inside the Frame*/
        PMACC_ALIGN(idx, uint32_t);

        /** pointer to parent frame where this particle is from
         *
         * ATTENTION: The pointer must be the last member to avoid local memory usage
         *            https://github.com/ComputationalRadiationPhysics/picongpu/pull/762
         */
        PMACC_ALIGN(frame, FrameType*);

        /** set particle handle to invalid
         *
         * This method sets the particle handle to invalid. It is possible to test with
         * the method isHandleValid if the particle is valid.
         * If the particle is set to invalid it is not allowed to call any method other
         * than isHandleValid or setHandleInvalid, but it does not mean the particle is
         * deactivated outside of this instance.
         */
        HDINLINE void setHandleInvalid()
        {
            frame = nullptr;
        }

        /** check if particle handle is valid
         *
         * A valid particle handle means that the memory behind the handle can be used
         * savely. A valid handle does not mean that the particle's multiMask is valid (>=1).
         *
         * @return true if the particle handle is valid, else false
         */
        HDINLINE bool isHandleValid() const
        {
            return frame != nullptr;
        }

        /** create particle
         *
         * @param frame reference to parent frame
         * @param idx index of particle inside the frame
         */
        HDINLINE Particle(FrameType& frame, uint32_t idx) : frame(&frame), idx(idx)
        {
        }

        template<typename T_OtherParticle>
        HDINLINE Particle(const T_OtherParticle& other) : frame(other.frame)
                                                        , idx(other.idx)
        {
        }

        /** access attribute with a identifier
         *
         * @param T_Key instance of identifier type
         *              (can be an alias, value_identifier or any other class)
         * @return result of operator[] of the Frame
         */
        template<typename T_Key>
        HDINLINE typename boost::result_of<
            typename boost::remove_reference<typename boost::result_of<FrameType(T_Key)>::type>::type(uint32_t)>::type
        operator[](const T_Key key)
        {
            PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);

            return frame->getIdentifier(key)[idx];
        }

        /** const version of method operator(const T_Key) */
        template<typename T_Key>
        HDINLINE typename boost::result_of<typename boost::remove_reference<
            typename boost::result_of<const FrameType(T_Key)>::type>::type(uint32_t)>::type
        operator[](const T_Key key) const
        {
            PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);

            return frame->getIdentifier(key)[idx];
        }

        HDINLINE
        ThisType& operator=(const ThisType& other) = default;

    private:
        /* we disallow to assign this class*/
        template<typename T_OtherParticle>
        HDINLINE ThisType& operator=(const T_OtherParticle& other);
    };

    namespace traits
    {
        template<typename T_Key, typename T_FrameType, typename T_ValueTypeSeq>
        struct HasIdentifier<pmacc::Particle<T_FrameType, T_ValueTypeSeq>, T_Key>
        {
        private:
            typedef pmacc::Particle<T_FrameType, T_ValueTypeSeq> ParticleType;
            typedef typename ParticleType::ValueTypeSeq ValueTypeSeq;

        public:
            /* If T_Key can not be found in the T_ValueTypeSeq of this Particle class,
             * SolvedAliasName will be void_.
             * Look-up is also valid if T_Key is an alias.
             */
            typedef typename GetKeyFromAlias<ValueTypeSeq, T_Key>::type SolvedAliasName;

            typedef bmpl::contains<ValueTypeSeq, SolvedAliasName> type;
        };

        template<typename T_Key, typename T_FrameType, typename T_ValueTypeSeq>
        struct HasFlag<pmacc::Particle<T_FrameType, T_ValueTypeSeq>, T_Key> : public HasFlag<T_FrameType, T_Key>
        {
        };

        template<typename T_Key, typename T_FrameType, typename T_ValueTypeSeq>
        struct GetFlagType<pmacc::Particle<T_FrameType, T_ValueTypeSeq>, T_Key>
            : public GetFlagType<T_FrameType, T_Key>
        {
        };

    } // namespace traits

    namespace particles
    {
        namespace operations
        {
            namespace detail
            {
                /** Assign common attributes of two particle species
                 *
                 * Assigns all attributes in ValueTypeSeq1 that also exist in T_ValueTypeSeq2
                 * from T_FrameType1 to T_FrameType2.
                 */
                template<
                    typename T_FrameType1,
                    typename T_ValueTypeSeq1,
                    typename T_FrameType2,
                    typename T_ValueTypeSeq2>
                struct Assign<
                    pmacc::Particle<T_FrameType1, T_ValueTypeSeq1>,
                    pmacc::Particle<T_FrameType2, T_ValueTypeSeq2>>
                {
                    typedef pmacc::Particle<T_FrameType1, T_ValueTypeSeq1> Dest;
                    typedef pmacc::Particle<T_FrameType2, T_ValueTypeSeq2> Src;

                    typedef typename Dest::ValueTypeSeq DestTypeSeq;
                    typedef typename Src::ValueTypeSeq SrcTypeSeq;

                    /* create attribute list with a subset of common attributes in two sequences
                     * bmpl::contains has lower complexity than traits::HasIdentifier
                     * and was used for this reason
                     */
                    typedef typename bmpl::copy_if<
                        DestTypeSeq,
                        bmpl::contains<SrcTypeSeq, bmpl::_1>,
                        bmpl::back_inserter<bmpl::vector0<>>>::type CommonTypeSeq;

                    /* create sequences with disjunct attributes from `DestTypeSeq` */
                    typedef typename bmpl::copy_if<
                        DestTypeSeq,
                        bmpl::not_<bmpl::contains<SrcTypeSeq, bmpl::_1>>,
                        bmpl::back_inserter<bmpl::vector0<>>>::type UniqueInDestTypeSeq;

                    /** Assign particle attributes
                     *
                     * The common subset of the attribute lists from both particles is
                     * used to set the attributes in dest with the corresponding ones from src.
                     * The remaining attributes that only exist in dest (UniqueInDestTypeSeq)
                     * are simply set to their default values.
                     *
                     * @param dest destination particle that shall be initialized/assigned with values from src
                     * @param src source particle were attributes are loaded from
                     */
                    HDINLINE
                    void operator()(Dest& dest, const Src& src)
                    {
                        using pmacc::meta::ForEach;
                        /* assign attributes from src to dest*/
                        ForEach<CommonTypeSeq, CopyIdentifier<bmpl::_1>> copy;
                        copy(dest, src);

                        /* set all attributes which are not in src to their default value*/
                        ForEach<UniqueInDestTypeSeq, SetAttributeToDefault<bmpl::_1>> setAttributeToDefault;
                        setAttributeToDefault(dest);
                    };
                };

                template<typename T_MPLSeqWithObjectsToRemove, typename T_FrameType, typename T_ValueTypeSeq>
                struct Deselect<T_MPLSeqWithObjectsToRemove, pmacc::Particle<T_FrameType, T_ValueTypeSeq>>
                {
                    typedef T_FrameType FrameType;
                    typedef T_ValueTypeSeq ValueTypeSeq;
                    typedef pmacc::Particle<FrameType, ValueTypeSeq> ParticleType;
                    typedef T_MPLSeqWithObjectsToRemove MPLSeqWithObjectsToRemove;

                    /* translate aliases to full specialized identifier*/
                    typedef typename ResolveAliases<
                        MPLSeqWithObjectsToRemove,
                        ValueTypeSeq,
                        errorHandlerPolicies::ReturnValue>::type ResolvedSeqWithObjectsToRemove;
                    /* remove types from original particle attribute list*/
                    typedef typename RemoveFromSeq<ValueTypeSeq, ResolvedSeqWithObjectsToRemove>::type NewValueTypeSeq;
                    /* new particle type*/
                    typedef pmacc::Particle<FrameType, NewValueTypeSeq> ResultType;

                    template<class>
                    struct result;

                    template<class F, class T_Obj>
                    struct result<F(T_Obj)>
                    {
                        typedef ResultType type;
                    };

                    HDINLINE
                    ResultType operator()(const ParticleType& particle)
                    {
                        return ResultType(particle);
                    };
                };

            } // namespace detail
        } // namespace operations
    } // namespace particles

} // namespace pmacc
