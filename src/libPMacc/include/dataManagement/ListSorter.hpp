/**
 * Copyright 2013-2016 Rene Widera, Felix Schmitt, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#include "dataManagement/IDataSorter.hpp"

#include <list>

namespace PMacc
{
    /**
     * Default sorter for DataConnector.
     * Uses a std::list for managing IDs.
     *
     * FIFO compliant.
     */
    template<typename ID_TYPE>
    class ListSorter : public IDataSorter<ID_TYPE>
    {
    public:
        ListSorter()
        {
            iter = ids.end();
        }

        ~ListSorter() {}

        void add(ID_TYPE id)
        {
            ids.push_back(id);
            if (iter == ids.end())
                iter = ids.begin();
        }

        ID_TYPE begin()
        {
            iter = ids.begin();
            return *iter;
        }

        bool isValid()
        {
            return iter != ids.end();
        }

        bool hasNext()
        {
            typename std::list<ID_TYPE>::iterator tmp_iter = iter;
            return (++tmp_iter) != ids.end();
        }

        ID_TYPE getNext()
        {
            iter++;
            return *iter;
        }

    private:
        std::list<ID_TYPE> ids;
        typename std::list<ID_TYPE>::iterator iter;
    };
}
