/* Copyright 2013-2022 Rene Widera, Alexander Grund
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

#include <boost/mp11.hpp>
#include <boost/mp11/mpl.hpp>
#include <boost/mpl/apply.hpp>

#include <llama/llama.hpp>

namespace pmacc
{
    namespace pmath = pmacc::math;

    namespace detail
    {
        template<typename PICValueType>
        using MakeLlamaField = llama::Field<PICValueType, typename traits::Resolve<PICValueType>::type::type>;

        template<typename ValueTypeSeq>
        using RecordDimFromValueTypeSeq = mp_rename<mp_transform<MakeLlamaField, ValueTypeSeq>, llama::Record>;

        template<typename T_MemoryLayout, typename SFINAE = void>
        inline constexpr bool splitVector = false;

        template<typename T_MemoryLayout>
        inline constexpr bool splitVector<T_MemoryLayout, std::void_t<decltype(T_MemoryLayout::splitVector)>>
            = T_MemoryLayout::splitVector;

        template<std::size_t T_size, typename T_ParticleDescription, typename T_MemoryLayout>
        struct ViewHolder
        {
        private:
            using IndexType = int; // TODO(bgruber): where do I get this type from?
            inline static constexpr IndexType particlesPerFrame
                = (T_size == llama::dyn) ? static_cast<IndexType>(llama::dyn) : static_cast<IndexType>(T_size);
            using RawRecordDim = RecordDimFromValueTypeSeq<typename T_ParticleDescription::ValueTypeSeq>;
            using SplitRecordDim = llama::TransformLeaves<RawRecordDim, pmath::ReplaceVectorByArray>;

        public:
            using RecordDim = std::conditional_t<splitVector<T_MemoryLayout>, SplitRecordDim, RawRecordDim>;
            using ArrayExtents = llama::ArrayExtents<IndexType, particlesPerFrame>;
            using Mapping = typename T_MemoryLayout::template fn<ArrayExtents, RecordDim>;
            static_assert(
                particlesPerFrame == llama::dyn || Mapping::blobCount == 1,
                "For statically sizes frames, only mappings with a single blob are supported");
            using BlobType = std::conditional_t<
                particlesPerFrame == llama::dyn,
                std::byte*,
                llama::Array<std::byte, Mapping{ArrayExtents{}}.blobSize(0)>>;
            using View = llama::View<Mapping, BlobType>;

        private:
            inline static constexpr std::size_t alignment
                = particlesPerFrame == llama::dyn ? alignof(std::byte*) : llama::alignOf<RecordDim>;

        public:
            alignas(alignment) View view;

            ViewHolder() = default;

            HDINLINE ViewHolder(IndexType size) : view{Mapping{ArrayExtents{size}}}
            {
            }

            HDINLINE auto& blobs()
            {
                return view.blobs();
            }

            HDINLINE auto blobSize(int i)
            {
                return view.mapping().blobSize(i);
            }
        };

        /** Proxy reference for particle attributes which are backed by a LLAMA RecordRef. This could become obsolete
         * when LLAMA's RecordRef supports operator= from TupleLike objects. Ask bgruber about it every now and then.
         */
        template<typename RecordRef>
        struct LlamaParticleAttribute
        {
            template<typename OtherRecordRef>
            auto operator=(const LlamaParticleAttribute<OtherRecordRef>& lpa) -> LlamaParticleAttribute&
            {
                rr = lpa.rr;
                return *this;
            }

            template<typename OtherRecordRef>
            auto operator=(LlamaParticleAttribute<OtherRecordRef>&& lpa) -> LlamaParticleAttribute&
            {
                rr = lpa.rr;
                return *this;
            }

            template<typename T>
            auto operator=(T&& t) -> LlamaParticleAttribute&
            {
                rr.store(std::forward<T>(t));
                return *this;
            }

            template<typename T>
            operator T() const
            {
                return rr.template loadAs<T>();
            }

            RecordRef rr;
        };
    } // namespace detail

    /** Frame is a storage for arbitrary number >0 of Particles with attributes
     *
     * @tparam T_size Static number of particles this frame stores, or llama::dyn for dynamic size
     * @tparam T_ValueTypeSeq sequence with value_identifier
     * @tparam T_MethodsList sequence of classes with particle methods
     *                       (e.g. calculate mass, gamma, ...)
     * @tparam T_Flags sequence with identifiers to add flags on a frame
     *                 (e.g. useSolverXY, calcRadiation, ...)
     * @tparam T_MemoryLayout Memory layout to be used for the particle attribute data.
     */
    template<std::size_t T_size, typename T_ParticleDescription, typename T_MemoryLayout>
    struct Frame;

    template<std::size_t T_size, typename T_ParticleDescription, typename T_MemoryLayout>
    struct Frame
        : public detail::ViewHolder<T_size, T_ParticleDescription, T_MemoryLayout>
        , public InheritLinearly<mp_append<
              typename T_ParticleDescription::MethodsList,
              typename OperateOnSeq<
                  typename T_ParticleDescription::FrameExtensionList,
                  boost::mpl::apply1<boost::mpl::_1, Frame<T_size, T_ParticleDescription, T_MemoryLayout>>>::type>>
    {
        using ViewHolder = detail::ViewHolder<T_size, T_ParticleDescription, T_MemoryLayout>;
        static_assert(
            T_size == llama::dyn
            || sizeof(ViewHolder) ==
                typename ViewHolder::Mapping{llama::ArrayExtents<int, static_cast<int>(T_size)>{}}.blobSize(0));

        using ViewHolder::ViewHolder;

        using ParticleDescription = T_ParticleDescription;
        using Name = typename ParticleDescription::Name;
        using SuperCellSize = typename ParticleDescription::SuperCellSize;
        using ValueTypeSeq = typename ParticleDescription::ValueTypeSeq;
        using MethodsList = typename ParticleDescription::MethodsList;
        using FlagList = typename ParticleDescription::FlagsList;
        using FrameExtensionList = typename ParticleDescription::FrameExtensionList;
        using ThisType = Frame;

        /* type of a single particle*/
        using ParticleType = pmacc::Particle<ThisType>;

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

    private:
        template<typename Frame, typename T_Key>
        static HDINLINE decltype(auto) at(Frame& f, uint32_t i, const T_Key key)
        {
            using Key = typename GetKeyFromAlias<ValueTypeSeq, T_Key, errorHandlerPolicies::ThrowValueNotFound>::type;
            auto&& ref = f.view(i)(Key{});

            using OldDstType = typename traits::Resolve<Key>::type::type;
            using RefType = std::remove_reference_t<decltype(ref)>;

            if constexpr(pmath::isVector<OldDstType> && llama::isRecordRef<RefType>)
                return pmath::makeVectorWithLlamaStorage<OldDstType>(ref);
            else if constexpr(llama::isRecordRef<RefType>)
                return detail::LlamaParticleAttribute<RefType>{ref};
            else
                return ref;
        }

    public:
        template<typename T_Key>
        HDINLINE decltype(auto) get(uint32_t i, const T_Key)
        {
            return at(*this, i, T_Key{});
        }

        template<typename T_Key>
        HDINLINE decltype(auto) get(uint32_t i, const T_Key) const
        {
            return at(*this, i, T_Key{});
        }

        HINLINE static std::string getName()
        {
            return Name::str();
        }
    };

    namespace traits
    {
        template<
            typename T_IdentifierName,
            std::size_t T_size,
            typename T_ParticleDescription,
            typename T_MemoryLayout>
        struct HasIdentifier<pmacc::Frame<T_size, T_ParticleDescription, T_MemoryLayout>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_size, T_ParticleDescription, T_MemoryLayout>;

        public:
            using ValueTypeSeq = typename FrameType::ValueTypeSeq;
            /* if T_IdentifierName is void_ than we have no T_IdentifierName in our Sequence.
             * check is also valid if T_Key is a alias
             */
            using SolvedAliasName = typename GetKeyFromAlias<ValueTypeSeq, T_IdentifierName>::type;

            using type = boost::mp11::mp_contains<ValueTypeSeq, SolvedAliasName>; // FIXME(bgruber): boost::mp11::
                                                                                  // needed because of nvcc 11.0 bug
        };

        template<
            typename T_IdentifierName,
            std::size_t T_size,
            typename T_ParticleDescription,
            typename T_MemoryLayout>
        struct HasFlag<pmacc::Frame<T_size, T_ParticleDescription, T_MemoryLayout>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_size, T_ParticleDescription, T_MemoryLayout>;
            using SolvedAliasName = typename pmacc::traits::GetFlagType<FrameType, T_IdentifierName>::type;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = mp_contains<FlagList, SolvedAliasName>;
        };

        template<
            typename T_IdentifierName,
            std::size_t T_size,
            typename T_ParticleDescription,
            typename T_MemoryLayout>
        struct GetFlagType<pmacc::Frame<T_size, T_ParticleDescription, T_MemoryLayout>, T_IdentifierName>
        {
        private:
            using FrameType = pmacc::Frame<T_size, T_ParticleDescription, T_MemoryLayout>;
            using FlagList = typename FrameType::FlagList;

        public:
            using type = typename GetKeyFromAlias<FlagList, T_IdentifierName>::type;
        };

    } // namespace traits

} // namespace pmacc
