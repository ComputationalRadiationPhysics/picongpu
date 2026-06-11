/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Benjamin Worpitz, Alexander Grund, Franz Poeschel, Pawel Ordyna, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/multi/IHelp.hpp"

#    include <memory>

namespace picongpu
{
    namespace openPMD
    {
        std::shared_ptr<plugins::multi::IHelp> getOpenPMDWriterHelp();
    } // namespace openPMD
} // namespace picongpu

#endif
