/* Copyright 2021-2024 Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#if ENABLE_OPENPMD == 1

#    include "picongpu/plugins/openPMD/Json.hpp"

#    include <openPMD/auxiliary/JSON.hpp>
#    include <openPMD/openPMD.hpp>

/*
 * Note:
 * This is a hostonly .cpp file because CMake will not use -isystem for system
 * include paths on NVCC targets created with alpaka_add_executable.
 * Since <nlohmann/json.hpp> throws a number of warnings, this .cpp file
 * ensures that NVCC never sees that library.
 */

namespace picongpu::openPMD
{
    auto resolveJsonConfig(std::string const& commandLineValue, MPI_Comm comm) -> std::string
    {
        /*
         * hdf5.dataset.chunks = none
         * --------------------------
         * Disable HDF5 chunking as it can conflict with MPI-IO backends.
         * This is very likely the same issue as
         * https://github.com/open-mpi/ompi/issues/7795.
         *
         *
         * adios2.engine.preferred_flush_target = "buffer"
         * -----------------------------------------------
         * Only relevant for ADIOS2 engines that support this feature,
         * ignored otherwise. Currently supported in BP5.
         * Small datasets should be written to the internal ADIOS2
         * buffer.
         * Big datasets should explicitly specify their flush target
         * in Series::flush(). Options are "buffer" and "disk".
         * Ideally, all flush() calls should specify this explicitly.
         *
         *
         * adios2.engine.parameters.BufferChunkSize = 2147381248
         * -----------------------------------------------------
         * This parameter is only interpreted by the ADIOS2 BP5 engine
         * (and potentially future engines that use the BP5 serializer).
         * Other engines will ignore it without warning.
         *
         * Reasoning: The internal data structure of BP5 is a linked
         * list of equally-sized chunks.
         * This parameter specifies the size of each individual chunk to
         * the maximum possible 2GB (i.e. a bit lower than that),
         * which is more performant than the default 128MB.
         *
         * Since each buffer chunk is allocated by malloc(), chunks are
         * not actually written upon creation.
         * As a result, on systems with virtual memory, the overhead
         * of specifying this is a potential allocation of up to 2GB
         * of unused **virtual** memory, **not physical** memory.
         * That's a good deal, since it gives us performance by default.
         *
         * On those systems where this setting actually poses a problem,
         * careful memory configuration is necessary anyway, so the
         * defaults don't matter.
         *
         * In openPMD >= 0.16, additionally ignore all attribute metadata
         * that comes from a rank other than rank 0.
         * The openPMD plugin of PIConGPU writes attributes synchronously
         * (since this is required in common configurations of HDF5).
         * For ADIOS2 however, it's better to write attributes from single
         * ranks only and avoid metadata duplication across ranks.
         * This option tells the ADIOS2 backend of openPMD that it can
         * safely ignore any attribute write if the current MPI rank is
         * not 0.
         */
        std::string const& baseConfigString = R"(
        {
          "hdf5": {
            "dataset": {
              "chunks": "none"
            }
          },
          "adios2": {
            "attribute_writing_ranks": 0,
            "engine": {
              "preferred_flush_target": "buffer",
              "parameters": {
                "BufferChunkSize": 2147381248
              }
            }
          }
        }
        )";
        return ::openPMD::json::merge(baseConfigString, commandLineValue, comm);
    }

} // namespace picongpu::openPMD

#endif // ENABLE_OPENPMD
