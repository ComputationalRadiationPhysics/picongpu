/* Copyright 2013-2023 Rene Widera
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

#include "pmacc/meta/ForEach.hpp"
#include "pmacc/meta/GetKeyFromAlias.hpp"
#include "pmacc/meta/conversion/RemoveFromSeq.hpp"
#include "pmacc/meta/conversion/ResolveAliases.hpp"
#include "pmacc/meta/errorHandlerPolicies/ReturnValue.hpp"
#include "pmacc/particles/boostExtension/InheritLinearly.hpp"
#include "pmacc/particles/operations/Assign.hpp"
#include "pmacc/particles/operations/CopyIdentifier.hpp"
#include "pmacc/particles/operations/Deselect.hpp"
#include "pmacc/particles/operations/SetAttributeToDefault.hpp"
#include "pmacc/static_assert.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/placeholders.hpp>

#include <type_traits>

#include <llama/llama.hpp>

namespace pmacc
{
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
        using FrameType = T_FrameType;
        using ValueTypeSeq = T_ValueTypeSeq;
        using Name = typename FrameType::Name;
        using SuperCellSize = typename FrameType::SuperCellSize;
        using ThisType = Particle<FrameType, ValueTypeSeq>;
        using MethodsList = typename FrameType::MethodsList;

        /** pointer to parent frame where this particle is from
         *
         * ATTENTION: The pointer must be the last member to avoid local memory usage
         *            https://github.com/ComputationalRadiationPhysics/picongpu/pull/762
         */
        PMACC_ALIGN(frame, FrameType*);

        /** index of particle inside the Frame*/
        PMACC_ALIGN(idx, uint32_t);

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
        HDINLINE decltype(auto) operator[](const T_Key key)
        {
            PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);
            return frame->get(idx, key);
        }

        /** const version of method operator(const T_Key) */
        template<typename T_Key>
        HDINLINE decltype(auto) operator[](const T_Key key) const
        {
            PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);
            return frame->get(idx, key);
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
            using ParticleType = pmacc::Particle<T_FrameType, T_ValueTypeSeq>;
            using ValueTypeSeq = typename ParticleType::ValueTypeSeq;

        public:
            /* If T_Key can not be found in the T_ValueTypeSeq of this Particle class,
             * SolvedAliasName will be void_.
             * Look-up is also valid if T_Key is an alias.
             */
            using SolvedAliasName = typename GetKeyFromAlias<ValueTypeSeq, T_Key>::type;

            using type = mp_contains<ValueTypeSeq, SolvedAliasName>;
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
                    using Dest = pmacc::Particle<T_FrameType1, T_ValueTypeSeq1>;
                    using Src = pmacc::Particle<T_FrameType2, T_ValueTypeSeq2>;

                    using DestTypeSeq = typename Dest::ValueTypeSeq;
                    using SrcTypeSeq = typename Src::ValueTypeSeq;

                    /* create sequences with disjunct attributes from `DestTypeSeq` */
                    using UniqueInDestTypeSeq = mp_set_difference<DestTypeSeq, SrcTypeSeq>;

                    /* create attribute list with a subset of common attributes in two sequences
                     * mp_contains has lower complexity than traits::HasIdentifier
                     * and was used for this reason
                     */
                    using CommonTypeSeq = mp_set_difference<DestTypeSeq, UniqueInDestTypeSeq>;

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
                        ForEach<CommonTypeSeq, CopyIdentifier<boost::mpl::_1>> copy;
                        copy(dest, src);

                        /* set all attributes which are not in src to their default value*/
                        ForEach<UniqueInDestTypeSeq, SetAttributeToDefault<boost::mpl::_1>> setAttributeToDefault;
                        setAttributeToDefault(dest);
                    };
                };

                template<typename T_MPLSeqWithObjectsToRemove, typename T_FrameType, typename T_ValueTypeSeq>
                struct Deselect<T_MPLSeqWithObjectsToRemove, pmacc::Particle<T_FrameType, T_ValueTypeSeq>>
                {
                    using FrameType = T_FrameType;
                    using ValueTypeSeq = T_ValueTypeSeq;
                    using ParticleType = pmacc::Particle<FrameType, ValueTypeSeq>;
                    using MPLSeqWithObjectsToRemove = T_MPLSeqWithObjectsToRemove;

                    /* translate aliases to full specialized identifier*/
                    using ResolvedSeqWithObjectsToRemove = typename ResolveAliases<
                        MPLSeqWithObjectsToRemove,
                        ValueTypeSeq,
                        errorHandlerPolicies::ReturnValue>::type;
                    /* remove types from original particle attribute list*/
                    using NewValueTypeSeq = typename RemoveFromSeq<ValueTypeSeq, ResolvedSeqWithObjectsToRemove>::type;
                    /* new particle type*/
                    using ResultType = pmacc::Particle<FrameType, NewValueTypeSeq>;

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
