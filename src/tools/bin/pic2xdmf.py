#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2021 Felix Schmitt, Conrad Schumann
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
import sys
import glob
import argparse
from xml.dom.minidom import Document
import splash2xdmf

doc = Document()
grid_doc = Document()
poly_doc = Document()
time_series = False

# identifiers for vector components
VECTOR_IDENTS = ["x", "y", "z", "w"]
NAME_DELIM = "/"

# PIC-specific names
NAME_GLOBALCELLIDX = "positionOffset"
NAME_POSITION = "position"


def get_vector_basename(vector_name):
    """
    Return the base name part for a vector attribute name
    """

    str_len = len(vector_name)

    if str_len < 3:
        return None

    for ident in VECTOR_IDENTS:
        if vector_name[str_len - 1] == ident and \
           vector_name[str_len - 2] == NAME_DELIM:
            return vector_name[0:(str_len - 2)]

    return None


def get_basegroup(name):
    """
    Return the base group part for an attribute name
    (without vector extension)
    """

    index = name.rfind(NAME_DELIM)
    if index == -1:
        return None

    return name[0:index]


def join_from_components(node_list, prefix, suffix, operation, dims):
    """
    Parameters:
    ----------------
    node_list: list of Elements
               nodes to join
    prefix:    string
               prefix of join operation, e.g. 'JOIN('
    suffix:    string
               suffix of join operation, e.g. ')'
    operation: string
               operator for join, e.g. '+'
    dims:      string
               dimensions of nodes in node_list
    Returns:
    ----------------
    return: Element
            new DataItem node
    """

    join_base = doc.createElement("DataItem")
    join_base.setAttribute("ItemType", "Function")
    join_base.setAttribute("Dimensions", "{} {}".format(dims, len(node_list)))

    # join components
    function_str = prefix

    index = 0
    for attr in node_list:
        join_base.appendChild(attr)
        function_str += "${}".format(index)
        if index < len(node_list) - 1:
            function_str += operation
        index += 1

    function_str += suffix
    join_base.setAttribute("Function", "{}".format(function_str))
    return join_base


def join_references_from_components(node_list, original_nodes_map, prefix,
                                    suffix, operation, dims):
    """
    Parameters:
    ----------------
    node_list: list of Elements
               nodes to join references of
    prefix:    string
               prefix of join operation, e.g. 'JOIN('
    suffix:    string
               suffix of join operation, e.g. ')'
    operation: string
               operator for join, e.g. '+'
    dims:      string
               dimensions of nodes in node_list
    Returns:
    ----------------
    return: Element
            new DataItem node with references to original nodes
    """

    join_base = doc.createElement("DataItem")
    join_base.setAttribute("ItemType", "Function")
    join_base.setAttribute("Dimensions", "{} {}".format(dims, len(node_list)))

    # join components
    function_str = prefix

    index = 0
    for attr in node_list:
        # create reference to attr node and append
        reference = doc.createElement("DataItem")
        reference.setAttribute("Reference", "XML")

        orig_name = original_nodes_map[attr].getAttribute("Name")
        extra_grid = ""
        if time_series:
            parent_node = original_nodes_map[attr].parentNode
            extra_grid = "Grid[@Name='{}']/".format(
                parent_node.getAttribute("Name"))

        node_text = ("/Xdmf/Domain/Grid/{}Attribute[@Name='{}']/"
                     "DataItem[1]").format(extra_grid, orig_name)
        reference_text = doc.createTextNode(node_text)
        reference.appendChild(reference_text)
        join_base.appendChild(reference)

        function_str += "${}".format(index)
        if index < len(node_list) - 1:
            function_str += operation
        index += 1

    function_str += suffix
    join_base.setAttribute("Function", "{}".format(function_str))
    return join_base


def create_vector_attribute_common(new_name, node_list):
    """
    Parameters:
    -----------
    new_name:  string
               name of the new vector Attribute node
    node_list: list of Elements
               XDMF DataItem elements to combine in a new vector Attribute
               node

    Returns:
    --------
    return: tuple
            vector_node, data_item_list, original_nodes_map, dims
    """

    vector_node = doc.createElement("Attribute")
    vector_node.setAttribute("Name", new_name)
    vector_node.setAttribute("AttributeType", "Vector")

    data_item_list = list()
    original_nodes_map = dict()

    for node in node_list:
        children = node.childNodes
        tmp_data_node = None
        tmp_info_nodes = list()

        for child in children:
            if child.nodeName == "Information":
                tmp_info_nodes.append(child.cloneNode(True))

            if child.nodeName == "DataItem":
                tmp_data_node = child.cloneNode(True)

        if tmp_data_node is None:
            print("Error: no DataItem found")

        for tmp_info_node in tmp_info_nodes:
            tmp_data_node.appendChild(tmp_info_node)

        data_item_list.append(tmp_data_node)
        original_nodes_map[tmp_data_node] = node

    dims = data_item_list[0].getAttribute("Dimensions")

    return (vector_node, data_item_list, original_nodes_map, dims)


def create_vector_attribute(new_name, node_list):
    """
    Parameters:
    -----------
    new_name:  string
               name of the new vector Attribute node
    node_list: list of Elements
               XDMF DataItem elements to combine in a new vector Attribute
               node

    Returns:
    --------
    return: Element
            new vector Attribute node
    """

    (vector_node, data_item_list, original_nodes_map, dims) = \
        create_vector_attribute_common(new_name, node_list)

    vector_data = join_from_components(data_item_list,
                                       "JOIN(", ")", ",", dims)
    vector_node.appendChild(vector_data)
    return vector_node


def create_vector_reference_attribute(new_name, node_list):
    """
    Parameters:
    ----------------
    new_name:  string
               name of the new vector Attribute node with DataItem references
    node_list: list of Elements
               XDMF DataItem elements to combine in a new vector Attribute
               node

    Returns:
    ----------------
    return: Element
            new vector Attribute node
    """

    (vector_node, data_item_list, original_nodes_map, dims) = \
        create_vector_attribute_common(new_name, node_list)

    vector_data = join_references_from_components(
        data_item_list, original_nodes_map, "JOIN(", ")", ",", dims)
    vector_node.appendChild(vector_data)
    return vector_node


def combine_positions(node_list, original_nodes_map, dims):
    """
    Combine position node using '+' operator

    Parameters:
    ----------------
    node_list: list of Elements
               position nodes to combine
    dims: string
          dimensions of nodes in node_list

    Returns:
    ----------------
    return: Element
            combined positions node
    """

    return join_references_from_components(node_list, original_nodes_map,
                                           "", "", "+", dims)


def create_position_geometry(node_list, dims):
    """
    Create a 2/3D Geometry XDMF node

    Parameters:
    ----------------
    node_list: list of Elements
               list of positions to join to a 2/3D geometry
    Returns:
    ----------------
    return: Element
            new Geometry XDMF node
    """

    geom_node = doc.createElement("Geometry")

    if len(node_list) == 2:
        geom_node.setAttribute("Type", "XY")
    else:
        geom_node.setAttribute("Type", "XYZ")

    combined_positions = join_from_components(node_list, "JOIN(", ")", ",",
                                              dims)

    geom_node.appendChild(combined_positions)
    return geom_node


def merge_grid_attributes(base_node):
    """
    Merge child attribute nodes of grid-type base_node as
    new vector attribute nodes if possible

    Parameters:
    ----------------
    base_node: Element
               base node (parent) for which children attributes shall be
               merged
    """

    vectors_map = dict()
    # sort all child attribute nodes of base_node into a map
    # according to their base name part
    for attr in base_node.getElementsByTagName("Attribute"):
        basename = get_vector_basename(attr.getAttribute("Name"))
        if basename is None:
            continue

        if basename not in vectors_map:
            vectors_map[basename] = [attr]
        else:
            vector_list = vectors_map.get(basename)
            vector_list.append(attr)

    # iterate over all map entries (basename, list of components/nodes)
    for (key, value_list) in vectors_map.items():
        # print("replacing nodes for basename {} with a {}-element " +
        #       vector".format(key, len(value_list)))
        vector_node = create_vector_reference_attribute(key, value_list)

        # uncomment to remove old component nodes from the xml tree
        # for old_attr in value_list:
        #     base_node.removeChild(old_attr)

        base_node.appendChild(vector_node)


def merge_poly_attributes(base_node):
    """
    Merge child attribute nodes of poly-type base_node as
    new vector attribute nodes if possible, combine geometry attributes

    Parameters:
    ----------------
    base_node: Element
               base node (parent) for which children attributes shall be
               merged
    """

    vectors_map = dict()
    # sort all child attribute nodes of base_node into a map
    # according to their base group part and further to their base name part
    # this creates a map (basegroup name) of maps (non_vector_name) of lists
    # (attribute nodes)
    for attr in base_node.getElementsByTagName("Attribute"):
        attr_name = attr.getAttribute("Name")
        non_vector_name = get_vector_basename(attr_name)
        if non_vector_name is None:
            non_vector_name = attr_name

        basegroup = get_basegroup(non_vector_name)
        if basegroup is None:
            continue

        if basegroup not in vectors_map:
            group_map = dict()
            group_map[non_vector_name] = [attr]
            vectors_map[basegroup] = group_map
        else:
            group_map = vectors_map.get(basegroup)
            if non_vector_name in group_map:
                group_map[non_vector_name].append(attr)
            else:
                group_map[non_vector_name] = [attr]

    # iterate over base group
    for (groupName, groupMap) in vectors_map.items():
        pos_vector_list = None
        gcellidx_vector_list = None
        number_of_elements = 0

        for (vectorName, vectorAttrs) in groupMap.items():
            if vectorName.endswith("/{}".format(NAME_POSITION)):
                pos_vector_list = vectorAttrs
                for i in pos_vector_list:
                    for child in i.childNodes:
                        if child.nodeName == "DataItem":
                            number_of_elements = \
                                child.getAttribute("Dimensions")
            else:
                if vectorName.endswith("/{}".format(NAME_GLOBALCELLIDX)):
                    gcellidx_vector_list = vectorAttrs
                else:
                    if len(vectorAttrs) > 1:
                        # print("replacing nodes for basename {} with a " +
                        #       "{}-element vector".format(
                        #       vectorName, len(vectorAttrs)))
                        vector_node = create_vector_reference_attribute(
                            vectorName, vectorAttrs)

                        # uncomment to remove old component nodes from the
                        # xml tree
                        # for attr in vectorAttrs:
                        #     base_node.removeChild(attr)

                        base_node.appendChild(vector_node)

        # now check that we have NAME_GLOBALCELLIDX and NAME_POSITION and
        # they match
        if gcellidx_vector_list is None:
            print("Error: Did not find attributes '{}' in group '{}'".format(
                  NAME_GLOBALCELLIDX, groupName))
            return

        if pos_vector_list is None:
            print("Error: Did not find attributes '{}' in group '{}'".format(
                  NAME_POSITION, groupName))
            return

        if len(gcellidx_vector_list) < 2 or len(gcellidx_vector_list) > 3:
            print(("Error: Attributes for '{}' in group '{}' are not a 2/3 "
                   "component vector").format(NAME_GLOBALCELLIDX, groupName))
            return

        if len(gcellidx_vector_list) != len(pos_vector_list):
            print(("Error: Vectors for '{}' and '{}' in group '{}' do not "
                   "match").format(NAME_GLOBALCELLIDX, NAME_POSITION,
                                   groupName))
            return

        combined_pos_nodes = list()
        for i in range(len(pos_vector_list)):
            pos_data_item = None
            gcell_data_item = None
            original_nodes_map = dict()

            for v_data in gcellidx_vector_list[i].childNodes:
                if v_data.nodeName == "DataItem":
                    gcell_data_item = v_data.cloneNode(True)
                    original_nodes_map[gcell_data_item] = v_data.parentNode
                    break

            for p_data in pos_vector_list[i].childNodes:
                if p_data.nodeName == "DataItem":
                    pos_data_item = p_data.cloneNode(True)
                    original_nodes_map[pos_data_item] = p_data.parentNode
                    break

            combined_node = combine_positions(
                [gcell_data_item, pos_data_item],
                original_nodes_map, number_of_elements)
            combined_pos_nodes.append(combined_node)

        geom_node = create_position_geometry(combined_pos_nodes,
                                             number_of_elements)
        base_node.appendChild(geom_node)

        # uncomment to remove old nodes
        # for (vectorName, vectorAttrs) in groupMap.items():
        #     if vectorName.endswith("/{}".format(NAME_GLOBALCELLIDX)) or \
        #        vectorName.endswith("/{}".format(NAME_POSITION)):
        #         for node in vectorAttrs:
        #             base_node.removeChild(node)


def transform_xdmf_xml(root):
    """
    Transform XDMF XML tree starting at node root

    Parameters:
    ----------------
    root: Element
          XML root element
    """

    for grid_node in root.getElementsByTagName("Grid"):
        if grid_node.getAttribute("GridType") == "Uniform":
            if grid_node.getAttribute("Name").startswith("Grid_"):
                merge_grid_attributes(grid_node)

            if grid_node.getAttribute("Name").startswith("Poly_"):
                merge_poly_attributes(grid_node)


# program functions

def get_args_parser():
    """
    Return the argument parser for this script

    Returns:
    ----------------
    return: ArgumentParser
            command line args parser
    """

    parser = argparse.ArgumentParser(description="Create a PIConGPU XDMF " +
                                     "meta description file from a " +
                                     "libSplash HDF5 file.")

    parser.add_argument("splashfile", metavar="<filename>",
                        help="libSplash HDF5 file with domain information")

    parser.add_argument("-o", metavar="<filename>", help="Name of output " +
                        "XDMF file (default: append '.xmf')")

    parser.add_argument("-t", "--time", help="Aggregate information over a " +
                        "time-series of libSplash data", action="store_true")

    parser.add_argument("--fullpath",
                        help="Use absolute paths for HDF5 files",
                        action="store_true")

    parser.add_argument("--no_splitgrid",
                        help="Avoid the XML-tree to be split in grid and " +
                             "poly grids for seperate output files",
                        action="store_true")

    return parser


def main():
    """
    Main
    """
    global time_series

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
        tmp = splashFilename.rfind(".h5")
        splashFilename = splashFilename[:tmp]

    output_filename = "{}.xmf".format(splashFilename)

    if args.o:
        if args.o.endswith(".xmf"):
            output_filename = args.o
        else:
            print("The script was stopped, because your output filename " +
                  "doesn't have\nan ending paraview can work with. " +
                  "Please use the ending '.xmf'!")
            sys.exit()

    if args.no_splitgrid:
        # create the basic xml structure using splash2xdmf
        xdmf_root = splash2xdmf.create_xdmf_xml(splash_files, args)
        # transform this xml using our pic semantic knowledge
        transform_xdmf_xml(xdmf_root)
        # create a xml file from the transformed structure
        doc.appendChild(xdmf_root)
        # write data to output file
        splash2xdmf.write_xml_to_file(output_filename, doc)
    else:
        # create the basic xml structure for grid and poly using splash2xdmf
        grid_xdmf_root, poly_xdmf_root = splash2xdmf.create_xdmf_xml(
            splash_files, args)
        # transform these xml using our pic semantic knowledge
        transform_xdmf_xml(grid_xdmf_root)
        transform_xdmf_xml(poly_xdmf_root)
        # create the xml files from the transformed structures
        grid_doc.appendChild(grid_xdmf_root)
        poly_doc.appendChild(poly_xdmf_root)
        # create list of two output filenames with _grid and _poly extension
        output_filename_list = \
            splash2xdmf.handle_user_filename(output_filename)
        # use output filename list to write data to the relating output file
        for output_file in output_filename_list:
            if output_file.endswith("_grid.xmf"):
                splash2xdmf.write_xml_to_file(output_file, grid_doc)
            if output_file.endswith("_poly.xmf"):
                splash2xdmf.write_xml_to_file(output_file, poly_doc)


if __name__ == "__main__":
    main()
