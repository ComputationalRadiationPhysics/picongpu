/**
 * Copyright 2013-2016 Benjamin Schneider, Rene Widera, Axel Huebl
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

#ifndef MESSAGE_IDS_HPP
#define MESSAGE_IDS_HPP

/** Possible message IDs */
enum MessageID : uint32_t
{
    NoMessage = 1u,             // if no message was received
    Image,                      // if the message is an image
    ImageSize,                  // the message contains two uints specifiing width and height of the image
    GridSize,                   // the message contains three uints giving the grid cells in X, Y and Z
    TimeStep,                   // the message contains one uint telling us the current time step
    VisName,                    // the message contains a string (chars) giving the name of the simulation
    CameraPosition,             // three floats determining the camera position
    CameraOrbit,                // three floats determining a camera movement vector
    CameraPan,                  // three floats determining orbiting the focal point around the camera position
    CameraSlide,                // three floats moving the camera and focal point at the same time
    CameraFocalPoint,           // three float determining the look at point of the camera
    CameraDefault,              // message telling to reset the camera view to default
    Weighting,                  // one float telling the weighting of the two data sources
    SimPlay,                    // run the simulation
    SimPause,                   // pause the simulation and only visualize
    CloseConnection,            // empty message telling us that the connection is about to be closed
    TransferFunctionA,          // an array of TF_RESOLUTION float4s describing a color lookup table
    TransferFunctionB,          // an array of TF_RESOLUTION float4s describing a color lookup table
    Clipping,                   // six floats in range ]0;1[ defining a clipping box with min x,y,z and max x,y,z
    BackgroundColor,            // three floats that carry the RGB values in range ]0;1[ determining the background color of the final image
    AvailableDataSource,        // a string containing the name of an available data source
    DataSourceA,                // a string setting the first data source to be visualized by name
    DataSourceB,                // a string setting the second data source to be visualized by name
    RequestDataSources,         // send a token to ask the simulation which datasources it provides
    ListVisualizations,         // token message used to query the visualization server for a list of available visualizations
    VisListLength,              // Number of visualizations in the previously requested list
    VisRivURI,                  // URI of a visualizations to connect to via RivLib
    CompositingModeAlphaBlend,
    CompositingModeIsoSurface,
    CompositingModeMIP,
    VisibleSimulationArea,      // six floats describing the XYZ min and XYZ max coordinates of the visible simulation area in world space
    IsoSurfaceValue,            // the normalized value defining the iso surface for the Isosurface compositing mode, one float in range ]0;1[
    PngWriterOn,                // message token (zero byte message) to start writing the simulation images to disk
    PngWriterOff,                // stop writing PNG images
    FPS,
    RenderFPS,
    NumGPUs,                    // number of simulating GPUs
    NumCells,                   // number of simulated cells (global)
    NumParticles                // number of simulated particles (global)
};

#endif /* MESSAGE_IDS_HPP */
