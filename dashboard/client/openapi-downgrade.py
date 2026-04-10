import argparse
import json
import sys

from enum import Enum

DESCRIPTION = """
This program attempts to downgrade OpenAPI 3.1 specification files to version 3.0.3, so that they can be used with common generators that still lack full support for 3.1,
including the popular openapi-generator (https://openapi-generator.tech). It may not work in all cases and was only tested with a number of FastAPI projects.

It provides workarounds for the following incompatibilities:
  - const definitions       -> enum with single value
  - null types              -> remove and set the nullable property for siblings

"""

REDOC_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <title>My Project - ReDoc</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="shortcut icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
    <style data-styled="" data-styled-version="4.4.1"></style>
</head>
<body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"> </script>
    <script>
        var spec = %s;
        Redoc.init(spec, {}, document.getElementById("redoc-container"));
    </script>
</body>
</html>
"""

class JSONType(Enum):
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"
    ARRAY = "array"
    BOOLEAN = "boolean"
    NULL = "null"

class Node:
    """
    Represents a node in a tree structure.

    Attributes:
    key -- The key associated with this node
    value -- The value associated with this node
    parent -- The parent of this node (None for the root)
    dt -- The data type of this node (defaults to JSONType.OBJECT)
    """
    def __init__(self, key, value, parent=None, dt=JSONType.OBJECT):
        self.key = key
        self.value = value
        self.parent = parent
        self.children = []
        self.dt = dt

    def __repr__(self):
        return f"Node({self.dt}, {self.key}={self.value}, {[c for c in self.children]})"

def json_to_tree(key, json_obj, parent=None):
    """
    Converts a JSON object into a tree structure.

    Args:
        key (str): The key of the current JSON object.
        json_obj (dict or list or str or int or float or bool or None): The JSON object to convert.
        parent (Node, optional): The parent node in the tree structure. Defaults to None.

    Returns:
        Node: The root node of the converted tree structure.
    """
    node = Node(key, json_obj, parent)
    if parent is not None:
        parent.children.append(node)
    if isinstance(json_obj, dict):
        node.dt = JSONType.OBJECT
        for k, v in json_obj.items():
            json_to_tree(k, v, node)
    elif isinstance(json_obj, list):
        node.dt = JSONType.ARRAY
        for i, item in enumerate(json_obj):
            json_to_tree(i, item, node)
    else:
        node.dt = JSONType.STRING
    return node


def tree_to_json(node):
    """
    Converts a tree node to a JSON object.

    Args:
        node (TreeNode): The tree node to convert.

    Returns:
        dict or list or any: The JSON representation of the tree node.
    """
    if node.dt == JSONType.OBJECT:
        return {child.key: tree_to_json(child) for child in node.children}
    elif node.dt == JSONType.ARRAY:
        return [tree_to_json(child) for child in node.children]
    else:
        return node.value


def visit_nodes(node, key=None, callback=None, level=0, depth=-1):
    """
    Recursively visits the nodes in a tree-like structure.

    Args:
        node: The current node to visit.
        key: The key to match against the node's key. If None, all nodes are visited.
        callback: The function to call for each visited node.
        level: The current level in the tree-like structure.
        depth: The maximum depth to visit. If -1, there is no depth limit.

    Returns:
        None
    """
    if node.key == key or key is None:
        if callback is not None:
            callback(node, level)
    for child in node.children:
        visit_nodes(child, key, callback, level+1, depth)


def add_nullable(node, _):
    """
    Adds the 'nullable' property to the child nodes if any of them have a type of 'null'.
    Removes the child nodes with a type of 'null'.

    Args:
        node (Node): The parent node.
        _ (Any): Placeholder argument.

    Returns:
        None
    """
    if any(isinstance(child.value, dict) and child.value.get('type') == 'null' for child in node.children):
        for child in node.children:
            if isinstance(child.value, dict) and child.value.get('type') != 'null':
                child.children.append(Node('nullable', True, child, JSONType.BOOLEAN))
            if child.value.get('type') == 'null':
                node.children.remove(child)


def flatten_anyof(node, _):
    """
    Flattens 'anyOf' nodes that have only one child by promoting the child's properties
    to the parent level.
    
    Args:
        node (Node): The anyOf node to potentially flatten.
        _ (Any): Placeholder argument.
    
    Returns:
        None
    """
    if len(node.children) == 1:
        single_child = node.children[0]
        
        if isinstance(single_child.value, dict) and single_child.dt == JSONType.OBJECT:
            parent = node.parent
            if parent:
                parent.children.remove(node)
                for grandchild in single_child.children:
                    grandchild.parent = parent
                    parent.children.append(grandchild)


def const_to_enum(node, _):
    """
    Convert a constant value to an enum.

    Args:
        node (Node): The node to convert.
        _ (Any): Placeholder argument.

    Returns:
        None
    """
    node.dt = JSONType.ARRAY
    node.key = "enum"
    node.children = [Node(0, node.value, node, JSONType.STRING)]
    node.value = [node.value]

def process_json(json_obj):
    """
    Process the given JSON object.

    Args:
        json_obj (dict): The JSON object to process.

    Returns:
        dict: The processed JSON object.
    """
    root = json_to_tree('root', json_obj)
    visit_nodes(root, "anyOf", add_nullable)
    visit_nodes(root, "anyOf", flatten_anyof)
    visit_nodes(root, "const", const_to_enum)
    return tree_to_json(root)


def main():

    parser = argparse.ArgumentParser(
        prog='oasdowngrade',
        description=DESCRIPTION,
    )
    parser.add_argument('filename')
    parser.add_argument('-o', '--outputfile', help='Output file name', default=None)
    parser.add_argument('-f', '--format', help='Output format (json, redoc)', action='store')
    args = parser.parse_args()

    try:
        with open(args.filename) as f:
            json_obj = json.load(f)
        
        converted_json = process_json(json_obj)

        if not args.outputfile:
            print(json.dumps(converted_json, indent=2))
            return

        fmt = args.format or "json"
        if fmt == 'redoc':
            output = REDOC_HTML_TEMPLATE % json.dumps(converted_json)
        elif fmt == 'json':
            output = json.dumps(converted_json)
        else:
            raise Exception('Invalid output format')

        if args.outputfile:
            with open(args.outputfile, 'w') as f:
                f.write(output)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
