/* Copyright 2013-2023 Rene Widera, Alexander Grund
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

#include "pmacc/math/MapTuple.hpp"
#include "pmacc/meta/GetKeyFromAlias.hpp"
#include "pmacc/meta/conversion/OperateOnSeq.hpp"
#include "pmacc/meta/conversion/SeqToMap.hpp"
#include "pmacc/particles/ParticleDescription.hpp"
#include "pmacc/particles/boostExtension/InheritLinearly.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/dataTypes/Particle.hpp"
#include "pmacc/traits/GetFlagType.hpp"
#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/traits/HasIdentifier.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/apply.hpp>

namespace pmacc
{
    namespace pmath = pmacc::math;

    /** Frame is a storage for arbitrary number >0 of Particles with attributes
     *
     * @tparam T_CreatePairOperator unary template operator to create a boost pair
     *                              from single type ( pair<name,dataType> )
     *                              @see MapTupel
     * @tparam T_ValueTypeSeq sequence with value_identifier
     * @tparam T_MethodsList sequence of classes with particle methods
     *                       (e.g. calculate mass, gamma, ...)
     * @tparam T_Flags sequence with identifiers to add flags on a frame
     *                 (e.g. useSolverXY, calcRadiation, ...)
     */
    template<typename T_CreatePairOperator, typename T_ParticleDescription>
    struct Frame;

    template<typename T_CreatePairOperator, typename T_ParticleDescription>
    struct Frame
        : protected pmath::MapTuple<
              typename SeqToMap<typename T_ParticleDescription::ValueTypeSeq, T_CreatePairOperator>::type>
        , public InheritLinearly<mp_append<
              typename T_ParticleDescription::MethodsList,
              typename OperateOnSeq<
                  typename T_ParticleDescription::FrameExtensionList,
                  boost::mpl::apply1<boost::mpl::_1, Frame<T_CreatePairOperator, T_ParticleDescription>>>::type>>
    {
        using ParticleDescription = T_ParticleDescription;
        using Name = typename ParticleDescription::Name;
        //! Number of particle slots within the frame
        using NumSlots = typename ParticleDescription::NumSlots;
        static constexpr uint32_t frameSize = NumSlots::value;
        using ValueTypeSeq = typename ParticleDescription::ValueTypeSeq;
        using MethodsList = typename ParticleDescription::MethodsList;
        using FlagList = typename ParticleDescription::FlagsList;
        using FrameExtensionList = typename ParticleDescription::FrameExtensionList;
        /* definition of the MapTupel where we inherit from*/
        using BaseType = pmath::MapTuple<typename SeqToMap<ValueTypeSeq, T_CreatePairOperator>::type>;

        /* type of a single particle*/
        using ParticleType = pmacc::Particle<Frame>;

        using SuperCellSize = typename ParticleDescription::SuperCellSize;
        static_assert(
            pmacc::math::CT::volume<SuperCellSize>::type::value <= frameSize,
            "Cells per supercell must be <= the number of particle slots in a frame!");

    public:
        /** access the Nth particle*/
        HDINLINE ParticleType operator[](const uint32_t idx)
        {
            return ParticleType(*this, idx);
        }

        /** access the Nth particle*/
        HDINLINE const ParticleType operator[](const uint32_t idx) const
        {
            return ParticleType(*this, idx);
        }

        /** access attribute with a identifier
         *
         * @param T_Key instance of identifier type
         *              (can be an alias, value_identifier or any other class)
         * @return result of operator[] of MapTupel
         */
        template<typename T_Key>
        HDINLINE auto& getIdentifier(const T_Key)
        {
            using Key = typename GetKeyFromAlias<ValueTypeSeq, T_Key, errorHandlerPolicies::ThrowValueNotFound>::type;
            return BaseType::operator[](Key());
        }

        /** const version of method getIdentifier(const T_Key) */
        template<typename T_Key>
        HDINLINE const auto& getIdentifier(const T_Key) const
        {
            using Key = typename GetKeyFromAlias<ValueTypeSeq, T_Key, errorHandlerPolicies::ThrowValueNotFound>::type;
            return BaseType::operator[](Key());
        }

        HINLINE static std::string getName()
        {
            return Name::str();
        }
    };

    namespace traits
    {
        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct HasIdentifier<pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>;

        public:
            using ValueTypeSeq = typename FrameType::ValueTypeSeq;
            /* if T_IdentifierName is void_ than we have no T_IdentifierName in our Sequence.
             * check is also valid if T_Key is a alias
             */
            using SolvedAliasName = typename GetKeyFromAlias<ValueTypeSeq, T_IdentifierName>::type;

            using type = boost::mp11::mp_contains<ValueTypeSeq, SolvedAliasName>; // FIXME(bgruber): boost::mp11::
                                                                                  // needed because of nvcc 11.0 bug
        };

        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct HasFlag<pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>;
            using SolvedAliasName = typename pmacc::traits::GetFlagType<FrameType, T_IdentifierName>::type;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = mp_contains<FlagList, SolvedAliasName>;
        };

        template<typename T_IdentifierName, typename T_CreatePairOperator, typename T_ParticleDescription>
        struct GetFlagType<pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_CreatePairOperator, T_ParticleDescription>;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = typename GetKeyFromAlias<FlagList, T_IdentifierName>::type;
        };

    } // namespace traits

} // namespace pmacc
