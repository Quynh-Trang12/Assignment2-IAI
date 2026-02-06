import sys
from graph import Graph
from engine import SearchEngine


def main():
    if len(sys.argv) < 3:
        print("Usage: python search.py <file> <method>")
        return

    filepath = sys.argv[1]
    method = sys.argv[2]

    # Check Valid Methods
    valid_methods = ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]
    if method.lower() not in valid_methods:
        print(f"Unknown method: {method}")
        return

    try:
        g = Graph()
        g.load_from_file(filepath)
    except Exception as e:
        print(f"Error: {e}")
        return

    engine = SearchEngine(g)
    res = engine.solve(method)

    print(f"{filepath} {method.upper()}")
    if res:
        print(f"{res[0]} {res[1]}")
        # path_str = str(res[2])
        # print(path_str)
        print(" ".join(map(str, res[2])))
    else:
        print("No solution found.")


