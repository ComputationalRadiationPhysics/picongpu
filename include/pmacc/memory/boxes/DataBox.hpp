/* Copyright 2013-2022 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
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

#include "SharedBox.hpp"
#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/shared/Allocate.hpp"

#include <llama/llama.hpp>

namespace pmacc
{
    namespace detail
    {
        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<1> const& idx = {})
        {
            return db[idx.x()];
        }

        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<2> const& idx = {})
        {
            return db[idx.y()][idx.x()];
        }

        template<typename DataBox>
        HDINLINE decltype(auto) access(const DataBox& db, DataSpace<3> const& idx = {})
        {
            return db[idx.z()][idx.y()][idx.x()];
        }
    } // namespace detail

    template<typename Base, typename SFINAE = void>
    struct DataBox : Base
    {
        HDINLINE DataBox() = default;

        HDINLINE DataBox(Base base) : Base{std::move(base)}
        {
        }

        HDINLINE DataBox(DataBox const&) = default;

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {}) const
        {
            /// @TODO(bgruber): inline and replace this by if constexpr in C++17 at some point. however, nvcc generates
            /// worse code with if constexpr. Ask Rene about it.
            return detail::access(*this, idx);
        }

        HDINLINE DataBox shift(DataSpace<Base::Dim> const& offset) const
        {
            DataBox result(*this);
            result.fixedPointer = &((*this)(offset));
            return result;
        }
    };

    namespace internal
    {
        template<typename... Sizes>
        HDINLINE constexpr auto toArrayExtents(math::CT::Vector<Sizes...>)
        {
            using V = math::CT::Vector<Sizes...>;
            using IndexType = typename math::CT::Vector<Sizes...>::type;
            if constexpr(V::dim == 1)
            {
                return llama::ArrayExtents<IndexType, V::x::value>{};
            }
            else if constexpr(V::dim == 2)
            {
                return llama::ArrayExtents<IndexType, V::y::value, V::x::value>{};
            }
            else if constexpr(V::dim == 3)
            {
                return llama::ArrayExtents<IndexType, V::z::value, V::y::value, V::x::value>{};
            }
            else
            {
                static_assert(sizeof(IndexType) == 0, "Vector dimension must be 1, 2 or 3");
            }
        }

        template<unsigned Dim>
        HDINLINE auto toArrayIndex(DataSpace<Dim> idx)
        {
            llama::ArrayIndex<typename DataSpace<Dim>::type, Dim> ai;
            for(int i = 0; i < Dim; i++)
                ai[i] = idx[Dim - 1 - i];
            return ai;
        }
    } // namespace internal

    // handle DataBox wrapping SharedBox with LLAMA
    template<typename T_TYPE, class T_SizeVector, typename T_MemoryMapping, uint32_t T_id, uint32_t T_dim>
    struct DataBox<
        SharedBox<T_TYPE, T_SizeVector, T_id, T_MemoryMapping, T_dim>,
        std::enable_if_t<!std::is_void_v<T_MemoryMapping>>>
    {
        using SharedBoxBase = SharedBox<T_TYPE, T_SizeVector, T_id, T_MemoryMapping, T_dim>;

        inline static constexpr std::uint32_t Dim = T_dim;
        using ValueType = T_TYPE;
        using Size = T_SizeVector;

        using SplitRecordDim = llama::TransformLeaves<T_TYPE, math::ReplaceVectorByArray>;
        using RecordDim = std::conditional_t<T_MemoryMapping::splitVector, SplitRecordDim, T_TYPE>;
        using ArrayExtents = decltype(internal::toArrayExtents(T_SizeVector{}));
        using Mapping = typename T_MemoryMapping::template fn<ArrayExtents, RecordDim>;
        using View = llama::View<Mapping, std::byte*>;

        View view;
        DataSpace<T_dim> offset{};

        HDINLINE DataBox() = default;

        HDINLINE DataBox(SharedBoxBase sb)
            : view{
                Mapping{{}},
                llama::Array<std::byte*, 1>{
                    const_cast<std::byte*>(reinterpret_cast<const std::byte*>(sb.fixedPointer))}}
        {
        }

        HDINLINE decltype(auto) operator()(DataSpace<T_dim> idx = {}) const
        {
            auto&& ref = const_cast<View&>(view)(internal::toArrayIndex(DataSpace<T_dim>{idx + offset}));
            if constexpr(math::isVector<T_TYPE> && llama::isRecordRef<std::remove_reference_t<decltype(ref)>>)
                return math::makeVectorWithLlamaStorage<T_TYPE>(ref);
            else
                return ref;
        }

        HDINLINE DataBox shift(const DataSpace<T_dim>& offset) const
        {
            // TODO(bgruber): can we enhance LLAMA to make this smarter than just keeping the offset?
            DataBox result(*this);
            result.offset += offset;
            return result;
        }

        template<typename T_Worker>
        static DINLINE SharedBoxBase init(T_Worker const& worker)
        {
            auto& mem_sh
                = memory::shared::allocate<T_id, memory::Array<ValueType, math::CT::volume<Size>::type::value>>(
                    worker);
            return {mem_sh.data()};
        }
    };
} // namespace pmacc