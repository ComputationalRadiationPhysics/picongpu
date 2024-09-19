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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
#include "pmacc/particles/operations/CopyValueIdentifier.hpp"
#include "pmacc/particles/operations/DeriveValueIdentifier.hpp"
#include "pmacc/particles/operations/Deselect.hpp"
#include "pmacc/particles/operations/InitValueIdentifier.hpp"
#include "pmacc/static_assert.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/traits/Resolve.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/plus.hpp>

#include <type_traits>

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
    private:
        /** Get the size in bytes for a value identifier */
        template<typename T_ValueIdentifier>
        struct GetSizeOfValueIdentifier
        {
            using ResolvedValueIdentifier = typename pmacc::traits::Resolve<T_ValueIdentifier>::type;
            using type = boost::mpl::integral_c<size_t, sizeof(typename ResolvedValueIdentifier::type)>;
        };

    public:
        using FrameType = T_FrameType;
        using ValueTypeSeq = T_ValueTypeSeq;
        using Name = typename FrameType::Name;

        // required to map local `cellIdx` to a cell within a supercell
        using SuperCellSize = typename FrameType::SuperCellSize;
        using MethodsList = typename FrameType::MethodsList;

        /** The sum of the bytes required to store all value identifiers (attributes) of a particle.
         *
         * @return size in bytes
         */
        static constexpr size_t sizeInByte()
        {
            namespace bmpl = boost::mpl;
            return bmpl::accumulate<
                ValueTypeSeq,
                bmpl::integral_c<size_t, 0u>,
                bmpl::plus<bmpl::_1, GetSizeOfValueIdentifier<bmpl::_2>>>::type::value;
        }

        /** pointer to parent frame where this particle is from
         *
         * ATTENTION: The pointer must be the last member to avoid local memory usage
         *            https://github.com/ComputationalRadiationPhysics/picongpu/pull/762
         */
        PMACC_ALIGN(frame, FrameType*) = nullptr;

        /** index of particle inside the Frame*/
        PMACC_ALIGN(idx, uint32_t) = std::numeric_limits<uint32_t>::max();

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

        HDINLINE Particle() = default;

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
        HDINLINE auto& operator[](const T_Key key)
        {
            PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);

            return frame->getIdentifier(key)[idx];
        }

        /** const version of method operator(const T_Key) */
        template<typename T_Key>
        HDINLINE const auto& operator[](const T_Key key) const
        {
            PMACC_CASSERT_MSG_TYPE(key_not_available, T_Key, traits::HasIdentifier<Particle, T_Key>::type::value);

            return frame->getIdentifier(key)[idx];
        }

        HDINLINE
        Particle& operator=(const Particle& other) = default;

        /** Derive attributes
         *
         * The common subset of the attribute lists from both particles is
         * used to set the attributes in this particle with the corresponding ones from source particle.
         * The remaining attributes that only exist in this particle is simply set to their default values.
         */
        template<typename T_Worker, typename T_OtherFrameType, typename T_OtherValueTypeSeq>
        HDINLINE void derive(
            T_Worker const& worker,
            IdGenerator& idGen,
            pmacc::Particle<T_OtherFrameType, T_OtherValueTypeSeq> const& srcParticle)
        {
            using DestTypeSeq = typename Particle::ValueTypeSeq;

            using pmacc::meta::ForEach;
            /* derive attributes */
            ForEach<DestTypeSeq, DeriveValueIdentifier<boost::mpl::_1>> derive;
            derive(worker, idGen, *this, srcParticle);
        }

        /** Assign common attributes of one particle to another
         *
         *  The source particle must have at least the attributes this particle has.
         */
        template<typename T_OtherFrameType, typename T_OtherValueTypeSeq>
        HDINLINE Particle& operator=(pmacc::Particle<T_OtherFrameType, T_OtherValueTypeSeq> const& other)
        {
            /* create sequences with disjunctive attributes */
            using UniqueInDestTypeSeq = mp_set_difference<ValueTypeSeq, T_OtherValueTypeSeq>;

            static_assert(
                pmacc::mp_size<UniqueInDestTypeSeq>::value == 0u,
                "Source particle is not providing all attributes required to update the destination particle.");

            using pmacc::meta::ForEach;
            /* derive attributes */
            ForEach<ValueTypeSeq, CopyValueIdentifier<boost::mpl::_1>> copy;
            copy(*this, other);
            return *this;
        }
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
