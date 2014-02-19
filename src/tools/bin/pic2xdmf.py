#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014 Felix Schmitt
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

import glob
import argparse
from xml.dom.minidom import Document
import splash2xdmf

doc = Document()

# identifiers for vector components
vector_idents = ["x", "y", "z", "w"]
vector_delim = "/"

# get the base name part for a vector attribute name
def get_vector_basename(vector_name):
    str_len = len(vector_name)
    
    if str_len < 3:
        return None
    
    for ident in vector_idents:
        if vector_name[str_len - 1] == ident and vector_name[str_len - 2] == vector_delim:
            return vector_name[0:(str_len - 2)]
        
    return None


# merge child attribute nodes of base_node as new vector attribute
# nodes if possible
def merge_attributes(base_node):
    vectors_map = dict()
    # sort all child attribute nodes of base_node into a map
    # according to their base name part
    for attr in base_node.getElementsByTagName("Attribute"):
        basename = get_vector_basename(attr.getAttribute("Name"))
        if basename == None:
            continue
            
        if not vectors_map.has_key(basename):
            vectors_map[basename] = [attr]
        else:
            vector_list = vectors_map.get(basename)
            vector_list.append(attr)
            
    # iterate over all map entries (basename, list of components/nodes)
    for (key, value_list) in vectors_map.items():
        print "replacing nodes for basename {} with a {}-element vector".format(key, len(value_list))
        vector_node = doc.createElement("Attribute")
        vector_node.setAttribute("Name", "{}".format(key))
        vector_node.setAttribute("AttributeType", "Vector")
        
        dims = value_list[0].firstChild.getAttribute("Dimensions")
        
        vector_data_base = doc.createElement("DataItem")
        vector_data_base.setAttribute("ItemType", "Function")
        vector_data_base.setAttribute("Dimensions", "{} {}".format(dims, len(value_list)))
        
        # join vector components
        function_str = "JOIN("
        
        index = 0
        for old_attr in value_list:
            vector_data_base.appendChild(old_attr.firstChild)
            # old component nodes are removed from the xml tree
            base_node.removeChild(old_attr)
            function_str += "${}".format(index)
            if index < len(value_list) - 1:
                function_str += ", "
            index += 1
            
        function_str += ")"
        vector_data_base.setAttribute("Function", "{}".format(function_str))
            
        vector_node.appendChild(vector_data_base)
        base_node.appendChild(vector_node)
            

def transform_xdmf_xml(root):
    for grid_node in root.getElementsByTagName("Grid"):
        if grid_node.getAttribute("GridType") == "Uniform":
            merge_attributes(grid_node)
    

# program functions

def get_args_parser():
    parser = argparse.ArgumentParser(description="Create a PIConGPU XDMF meta "
        "description file from a libSplash HDF5 file.")

    parser.add_argument("splashfile", metavar="<filename>",
        help="libSplash HDF5 file with domain information")
        
    parser.add_argument("-o", metavar="<filename>", help="Name of output XDMF "
        "file (default: append '.xmf')")
    
    parser.add_argument("-t", "--time", help="Aggregate information over a "
        "time-series of libSplash data", action="store_true")
        
    return parser


def main():
    # get arguments from command line
    args_parser = get_args_parser()
    args = args_parser.parse_args()

    # apply arguments
    splashFilename = args.splashfile
    time_series = args.time

    # create the list of requested splash files
    splash_files = list()
    if time_series:
        splashFilename = splash2xdmf.get_common_filename(splashFilename)
        for s_filename in glob.glob("{}_*.h5".format(splashFilename)):
            splash_files.append(s_filename)
    else:
        splash_files.append(splashFilename)
        
    # create the basic xml structure using splas2xdmf
    xdmf_root = splash2xdmf.create_xdmf_xml(splash_files)
    # transform this xml using our pic semantic knowledge
    transform_xdmf_xml(xdmf_root)

    # create a xml file from the transformed structure
    doc.appendChild(xdmf_root)
    
    output_filename = "{}.xmf".format(splashFilename)
    if args.o:
        output_filename = args.o
    splash2xdmf.write_xml_to_file(output_filename, doc)


if __name__ == "__main__":
    main()
