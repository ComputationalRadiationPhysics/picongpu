/* Copyright 2024 Julian Lenz
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

#include <pmacc/boost_workaround.hpp>

#include <pmacc/pluginSystem/IPlugin.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/value_semantic.hpp>

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

#include <nlohmann/json.hpp>


namespace picongpu
{
    namespace traits
    {
        // doc-include-start: GetMetdata trait
        template<typename TObject>
        struct GetMetadata
        {
            // for classes with compile-time information only, this can be left out:
            TObject const& obj;

            // for classes with compile-time information only, this can be left out:
            nlohmann::json descriptionRT() const
            {
                return obj.metadata();
            }

            // for classes with runtime-time information only, this can be left out:
            static nlohmann::json descriptionCT()
            {
                return TObject::metadata();
            }
        };
        // doc-include-end: GetMetdata trait
    } // namespace traits
} // namespace picongpu
