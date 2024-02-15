#Tatum Hansen CST 301
# Import the javalang module for parsing Java code and the json module for JSON operations.
import javalang
import json

# Define a function to parse Java code into an Abstract Syntax Tree (AST) using the javalang library.
def parse_java_code_to_ast(java_code):
    # Parse the Java code and return the AST.
    tree = javalang.parse.parse(java_code)
    return tree

# Define a function to print the AST in a hierarchical structure for readability.
def print_ast(node, prefix="", is_last=True):
    # Get the type name of the current node.
    node_type = type(node).__name__
    # Print the current node type with appropriate prefix and branch symbols.
    print(f"{prefix}{'└── ' if is_last else '├── '}{node_type}")

    # If the node has attributes, print each attribute and its value.
    if hasattr(node, 'attributes'):
        for attribute in node.attributes:
            value = getattr(node, attribute)
            # If the attribute value is a list, simplify its representation.
            if isinstance(value, list):
                value = f"[{', '.join(map(str, value))}]"
            print(f"{prefix}{'    ' if is_last else '|   '}├── {attribute}: {value}")

    # Get the children of the current node, if any.
    children = getattr(node, 'children', None) or []
    # Recursively print each child node.
    for i, child in enumerate(children):
        # Check if the child is a valid node and print it recursively.
        if isinstance(child, (javalang.tree.Node, dict)):  # Adjust based on your node types
            print_ast(child, prefix + ('    ' if is_last else '|   '), is_last=i == len(children) - 1)
        elif isinstance(child, list):
            # If the child is a list, iterate through its elements.
            for j, subchild in enumerate(child):
                if isinstance(subchild, (javalang.tree.Node, dict)):  # Adjust based on your node types
                    print_ast(subchild, prefix + ('    ' if is_last else '|   '), is_last=j == len(child) - 1)

# Define a function to convert the AST to a dictionary format.
def tree_to_dict(node):
    # Check if the node has attributes to include in the dictionary.
    if not hasattr(node, 'attributes'):
        return str(node)
    result = {"type": type(node).__name__, "attributes": {}}
    for attr in node.attributes:
        attr_value = getattr(node, attr)
        # Handle list of nodes by recursively converting them to dictionaries.
        if isinstance(attr_value, list):
            result["attributes"][attr] = [tree_to_dict(item) if isinstance(item, javalang.tree.Node) else item for item in attr_value]
        else:
            result["attributes"][attr] = attr_value

    # Process child nodes, if any, to include them in the dictionary.
    children = []
    for child in getattr(node, 'children', []):
        if isinstance(child, javalang.tree.Node):
            child_dict = tree_to_dict(child)
            if child_dict:  # Filter out None values.
                children.append(child_dict)
        elif isinstance(child, list):
            for c in child:
                if isinstance(c, javalang.tree.Node):
                    children.append(tree_to_dict(c))
    if children:
        result["children"] = children
    return result

# Define a function to write the AST dictionary to a JSON file.
def write_ast_to_json_file(tree_dict, filename):
    # Print the dictionary content in JSON format for verification.
    print("\nContent added to JSON:")
    print(json.dumps(tree_dict, indent=4))
    # Write the dictionary to a JSON file.
    with open(filename, 'w') as json_file:
        json.dump(tree_dict, json_file, indent=4)

# Sample Java code for demonstration.
java_code = """
public class HelloWorld 
{
    public static void main(String[] args) 
    {
        System.out.println("Hello, World!");
    }
}
"""

# Another sample Java code for demonstration.
java_code2 = """
import java.util.Random;

public class Addition 
{
    public static void main(String[] args) 
    {
        Random rand = new Random();
        int num1 = rand.nextInt(10); 
        int num2 = rand.nextInt(10);
        int sum = num1 + num2;
        System.out.println(sum);
    }
}
"""

# Parse and print the AST for the first sample code.
tree = parse_java_code_to_ast(java_code)
print("Printing 'Hello World' AST Tree:")
print_ast(tree)

# Convert the AST to a dictionary and write it to a JSON file.
tree_dict = tree_to_dict(tree)
write_ast_to_json_file(tree_dict, "ast.json")

# Parse and print the AST for the second sample code.
tree2 =  parse_java_code_to_ast(java_code2)
print("\nPrinting 'Random Addition' AST Tree:")
print_ast(tree2)

# Convert the AST of the second sample code to a dictionary and overwrite the previous JSON file.
tree_dict2 = tree_to_dict(tree2)
write_ast_to_json_file(tree_dict2, "ast.json")

print("\nAST has been written to ast.json")
