/* Copyright 2016-2021 Alexander Grund
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/adios/ADIOSWriter.def"
#include <pmacc/debug/VerboseLog.hpp>
#include "picongpu/traits/PICToAdios.hpp"
#include <adios.h>
#include <adios_read.h>
#include <adios_error.h>
#include <stdexcept>

namespace picongpu
{
    namespace adios
    {
        /**
         * Read an attribute from an open ADIOS file, check that its type is correct and return it
         *
         * @param fp       Open ADIOS file handle
         * @param basePath Path where the attribute is located in the file (with or w/o trailing slash)
         * @param attrName Name of the attribute. Used for status output and concatenated with basePath
         * @retval Attribute value
         */
        template<typename T_Attribute>
        T_Attribute readAttribute(ADIOS_FILE* fp, const std::string& basePath, const std::string& attrName)
        {
            // Build full path
            std::string attrPath = basePath;
            if(!attrPath.empty() && attrPath[attrPath.size() - 1] != '/')
                attrPath += '/';
            attrPath += attrName;
            // Actually read the data
            enum ADIOS_DATATYPES attrType;
            int attrSize;
            T_Attribute* attrValuePtr;
            ADIOS_CMD(adios_get_attr(fp, attrPath.c_str(), &attrType, &attrSize, (void**) &attrValuePtr));
            // Sanity checks
            if(attrType != traits::PICToAdios<T_Attribute>().type)
                throw std::runtime_error(std::string("Invalid type of ADIOS attribute ") + attrName);
            if(attrSize != sizeof(T_Attribute))
                throw std::runtime_error(std::string("Invalid size of ADIOS attribute ") + attrName);

            T_Attribute attribute = *attrValuePtr;
            __delete(attrValuePtr);
            log<picLog::INPUT_OUTPUT>("ADIOS: value of %1% = %2%") % attrName % attribute;
            return attribute;
        }

    } // namespace adios
} // namespace picongpu
