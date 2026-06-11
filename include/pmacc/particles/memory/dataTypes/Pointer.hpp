/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    /** Wrapper for a raw pointer that propagates its constness onto the pointee. Similar to
     * std::experimental::propagate_const<T>.
     *
     * @tparam T_Type type of the pointed object
     */
    template<typename T_Type>
    class Pointer
    {
    public:
        using type = T_Type;
        using PtrType = type*;
        using ConstPtrType = type const*;

        HDINLINE Pointer() = default;

        HDINLINE Pointer(PtrType const ptrIn) : ptr(ptrIn)
        {
        }

        HDINLINE Pointer(Pointer const& other) = default;

        HDINLINE Pointer& operator=(Pointer const& other) = default;

        /** dereference the pointer*/
        HDINLINE type& operator*()
        {
            return *ptr;
        }

        /** dereference the pointer*/
        HDINLINE const type& operator*() const
        {
            return *ptr;
        }

        /** access member*/
        HDINLINE PtrType operator->()
        {
            return ptr;
        }

        /** access member*/
        HDINLINE ConstPtrType operator->() const
        {
            return ptr;
        }

        /** compare if two pointers point to the same memory address*/
        HDINLINE bool operator==(Pointer<type> const& other) const
        {
            return ptr == other.ptr;
        }

        /** check if the memory address of two pointers are different*/
        HDINLINE bool operator!=(Pointer<type> const& other) const
        {
            return ptr != other.ptr;
        }

        /** check if the memory pointed to has a valid address
         * @return false if memory adress is nullptr else true
         */
        HDINLINE bool isValid() const
        {
            return ptr != nullptr;
        }

        PMACC_ALIGN(ptr, PtrType) { nullptr };
    };

} // namespace pmacc
