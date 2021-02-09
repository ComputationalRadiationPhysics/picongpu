/* Copyright 2013-2021 Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/cuSTL/container/allocator/tag.hpp"

#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/Environment.hpp"

#include <iostream>
#include <exception>
#include <sstream>


namespace pmacc
{
    namespace container
    {
        namespace detail
        {
            template<int dim>
            struct PitchHelper;

            template<>
            struct PitchHelper<1>
            {
                template<typename TCursor>
                HDINLINE math::Size_t<0u> operator()(const TCursor&)
                {
                    return math::Size_t<0u>();
                }

                HDINLINE math::Size_t<0u> operator()(const math::Size_t<1u>&)
                {
                    return math::Size_t<0u>();
                }
            };
            template<>
            struct PitchHelper<2>
            {
                template<typename TCursor>
                HDINLINE math::Size_t<1> operator()(const TCursor& cursor)
                {
                    return math::Size_t<1>(size_t((char*) cursor(0, 1).getMarker() - (char*) cursor.getMarker()));
                }

                HDINLINE math::Size_t<1> operator()(const math::Size_t<2>& size)
                {
                    return math::Size_t<1>(size.x());
                }
            };
            template<>
            struct PitchHelper<3>
            {
                template<typename TCursor>
                HDINLINE math::Size_t<2> operator()(const TCursor& cursor)
                {
                    return math::Size_t<2>(
                        (size_t)((char*) cursor(0, 1, 0).getMarker() - (char*) cursor.getMarker()),
                        (size_t)((char*) cursor(0, 0, 1).getMarker() - (char*) cursor.getMarker()));
                }

                HDINLINE math::Size_t<2> operator()(const math::Size_t<3>& size)
                {
                    return math::Size_t<2>(size.x(), size.x() * size.y());
                }
            };

            template<typename MemoryTag>
            HDINLINE void notifyEventSystem()
            {
            }

            template<>
            HDINLINE void notifyEventSystem<allocator::tag::device>()
            {
#ifndef __CUDA_ARCH__
                using namespace pmacc;
                __startOperation(ITask::TASK_DEVICE);
#endif
            }

            template<>
            HDINLINE void notifyEventSystem<allocator::tag::host>()
            {
#ifndef __CUDA_ARCH__
                using namespace pmacc;
                __startOperation(ITask::TASK_HOST);
#endif
            }
        } // namespace detail

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::CartBuffer(const math::Size_t<T_dim>& _size)
            : refCount(nullptr)
        {
            this->_size = _size;
            init();
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::CartBuffer(size_t x) : refCount(nullptr)
        {
            this->_size = math::Size_t<1>(x);
            init();
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::CartBuffer(size_t x, size_t y)
            : refCount(nullptr)
        {
            this->_size = math::Size_t<2>(x, y);
            init();
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::CartBuffer(size_t x, size_t y, size_t z)
            : refCount(nullptr)
        {
            this->_size = math::Size_t<3>(x, y, z);
            init();
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::CartBuffer(
            const CartBuffer<Type, T_dim, Allocator, Copier, Assigner>& other)
            : refCount(nullptr)
        {
            this->dataPointer = other.dataPointer;
            this->refCount = other.refCount;
            (*this->refCount)++;
            this->_size = other._size;
            this->pitch = other.pitch;
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::CartBuffer(
            CartBuffer<Type, T_dim, Allocator, Copier, Assigner>&& other)
            : refCount(nullptr)
        {
            this->dataPointer = other.dataPointer;
            this->refCount = other.refCount;
            this->_size = other._size;
            this->pitch = other.pitch;
            other.dataPointer = nullptr;
            other.refCount = nullptr;
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE void CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::init()
        {
            typename Allocator::Cursor cursor = Allocator::allocate(this->_size);
            this->dataPointer = cursor.getMarker();
#ifndef __CUDA_ARCH__
            this->refCount = new int;
            *this->refCount = 1;
#endif
            this->pitch = detail::PitchHelper<T_dim>()(cursor);
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::~CartBuffer()
        {
            exit();
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE void CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::exit()
        {
            if(!this->refCount)
                return;
            (*(this->refCount))--;
            if(*(this->refCount) > 0)
                return;
            Allocator::deallocate(origin());
            this->dataPointer = nullptr;
#ifndef __CUDA_ARCH__
            delete this->refCount;
            this->refCount = 0;
#endif
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>&
        CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::operator=(const CartBuffer& rhs)
        {
#ifndef __CUDA_ARCH__
            if(rhs.size() != this->size())
                throw std::invalid_argument(static_cast<std::stringstream&>(
                                                std::stringstream()
                                                << "Assignment: Sizes of buffers do not match: " << this->size()
                                                << " <-> " << rhs.size() << std::endl)
                                                .str());
#else
            assert(rhs.size() == this->size());
#endif

            if(this->dataPointer == rhs.dataPointer)
                return *this;
            Copier::copy(this->dataPointer, this->pitch, rhs.dataPointer, rhs.pitch, rhs._size);
            return *this;
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE CartBuffer<Type, T_dim, Allocator, Copier, Assigner>&
        CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::operator=(CartBuffer&& rhs)
        {
#ifndef __CUDA_ARCH__
            if(rhs.size() != this->size())
                throw std::invalid_argument(static_cast<std::stringstream&>(
                                                std::stringstream()
                                                << "Assignment: Sizes of buffers do not match: " << this->size()
                                                << " <-> " << rhs.size() << std::endl)
                                                .str());
#else
            assert(rhs.size() == this->size());
#endif
            if(this->dataPointer == rhs.dataPointer)
                return *this;

            exit();
            this->dataPointer = rhs.dataPointer;
            this->refCount = rhs.refCount;
            this->_size = rhs._size;
            this->pitch = rhs.pitch;
            rhs.dataPointer = nullptr;
            rhs.refCount = nullptr;
            return *this;
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE View<CartBuffer<Type, T_dim, Allocator, Copier, Assigner>>
        CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::view(math::Int<T_dim> a, math::Int<T_dim> b) const
        {
            a = (a + (math::Int<T_dim>) this->size()) % (math::Int<T_dim>) this->size();
            b = (b + (math::Int<T_dim>) this->size())
                % ((math::Int<T_dim>) this->size() + math::Int<T_dim>::create(1));

            View<CartBuffer<Type, T_dim, Allocator, Copier, Assigner>> result;

            result.dataPointer = &(*origin()(a));
            result._size = (math::Size_t<T_dim>) (b - a);
            result.pitch = this->pitch;
            result.refCount = this->refCount;
            return result;
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE cursor::BufferCursor<Type, T_dim> CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::origin() const
        {
            detail::notifyEventSystem<typename Allocator::tag>();
            return cursor::BufferCursor<Type, T_dim>(this->dataPointer, this->pitch);
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE cursor::SafeCursor<cursor::BufferCursor<Type, T_dim>>
        CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::originSafe() const
        {
            return cursor::make_SafeCursor(this->origin(), math::Int<T_dim>::create(0), math::Int<T_dim>(size()));
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<T_dim>, char*>
        CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::originCustomAxes(const math::UInt32<T_dim>& axes) const
        {
            math::Size_t<dim> factor;
            factor[0] = sizeof(Type);
            if(dim > 1)
                factor[1] = this->pitch[0];
            if(dim > 2)
                factor[2] = this->pitch[1];
            //\todo: is the conversation from size_t to int32_t allowed?
            math::Int<dim> customFactor;
            for(int i = 0; i < dim; i++)
                customFactor[i] = (int) factor[axes[i]];
            cursor::CartNavigator<dim> navi(customFactor);

            detail::notifyEventSystem<typename Allocator::tag>();

            return cursor::Cursor<cursor::PointerAccessor<Type>, cursor::CartNavigator<dim>, char*>(
                cursor::PointerAccessor<Type>(),
                navi,
                (char*) this->dataPointer);
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE zone::SphericZone<T_dim> CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::zone() const
        {
            zone::SphericZone<T_dim> myZone;
            myZone.offset = math::Int<T_dim>::create(0);
            myZone.size = this->_size;
            return myZone;
        }

        template<typename Type, int T_dim, typename Allocator, typename Copier, typename Assigner>
        HDINLINE bool CartBuffer<Type, T_dim, Allocator, Copier, Assigner>::isContigousMemory() const
        {
            return this->pitch == detail::PitchHelper<dim>()(this->_size);
        }

        template<typename Type, typename Allocator, typename Copier, typename Assigner>
        std::ostream& operator<<(std::ostream& s, const CartBuffer<Type, 1, Allocator, Copier, Assigner>& con)
        {
            for(size_t x = 0; x < con.size().x(); x++)
                s << con.origin()[x] << " ";
            return s << std::endl;
        }

        template<typename Type, typename Allocator, typename Copier, typename Assigner>
        std::ostream& operator<<(std::ostream& s, const CartBuffer<Type, 2, Allocator, Copier, Assigner>& con)
        {
            for(size_t y = 0; y < con.size().y(); y++)
            {
                for(size_t x = 0; x < con.size().x(); x++)
                    s << *con.origin()(x, y) << " ";
                s << std::endl;
            }
            return s << std::endl;
        }

        template<typename Type, typename Allocator, typename Copier, typename Assigner>
        std::ostream& operator<<(std::ostream& s, const CartBuffer<Type, 3, Allocator, Copier, Assigner>& con)
        {
            for(size_t z = 0; z < con.size().z(); z++)
            {
                for(size_t y = 0; y < con.size().y(); y++)
                {
                    for(size_t x = 0; x < con.size().x(); x++)
                        s << *con.origin()(x, y, z) << " ";
                    s << std::endl;
                }
                s << std::endl;
            }
            return s << std::endl;
        }

    } // namespace container
} // namespace pmacc
