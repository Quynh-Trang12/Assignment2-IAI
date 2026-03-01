import os
from typing import List, Dict, Tuple


class TestFactory:
    """
    TestFactory is a dedicated utility class responsible for generating the 10 distinct 
    topological test cases required for the Assignment 2A robustness and validation suite.
    
    It programmatically constructs graph definitions and writes them to disk using the 
    strict text-based formatting required by the SearchEngine's file parser.
    """

    # The relative directory path where all generated test case files will be stored.
    OUTPUT_DIRECTORY: str = "tests/cases"

    @staticmethod
    def ensure_output_directory_exists() -> None:
        """
        Safely verifies the existence of the target output directory on the filesystem.
        If it does not exist, it provisions the necessary folder structure.
        """
        if not os.path.exists(TestFactory.OUTPUT_DIRECTORY):
            os.makedirs(TestFactory.OUTPUT_DIRECTORY)

    @staticmethod
    def generate_test_file(
        filename: str, 
        nodes: Dict[int, Tuple[float, float]], 
        adjacency_list: Dict[int, Dict[int, float]], 
        origin: int, 
        destinations: List[int]
    ) -> None:
        """
        Serializes a graph's configuration into the specific text format required by the assignment.
        
        Args:
            filename (str): The name of the output text file (e.g., 'T01_Standard.txt').
            nodes (Dict[int, Tuple[float, float]]): A mapping of node identifiers to their (X, Y) spatial coordinates.
            adjacency_list (Dict[int, Dict[int, float]]): The directed edges mapping source nodes to target nodes and their weights.
            origin (int): The starting node identifier.
            destinations (List[int]): A list of valid goal node identifiers.
        """
        filepath = os.path.join(TestFactory.OUTPUT_DIRECTORY, filename)
        
        # Use a context manager with explicit UTF-8 encoding to guarantee cross-platform file writing
        with open(filepath, "w", encoding="utf-8") as file_stream:
            file_stream.write("Nodes:\n")
            for node_identifier, coordinates in nodes.items():
                file_stream.write(f"{node_identifier}: ({coordinates[0]},{coordinates[1]})\n")

            file_stream.write("Edges:\n")
            for source_node, neighbors in adjacency_list.items():
                for target_node, edge_weight in neighbors.items():
                    file_stream.write(f"({source_node},{target_node}): {edge_weight}\n")

            file_stream.write("Origin:\n")
            file_stream.write(f"{origin}\n")

            file_stream.write("Destinations:\n")
            # Enforce strict assignment formatting: destinations must be semi-colon separated
            formatted_destinations = "; ".join(map(str, destinations))
            file_stream.write(f"{formatted_destinations}\n")
            
        print(f"[TestFactory] Successfully provisioned test case: {filepath}")

    @classmethod
    def provision_all_test_cases(cls) -> None:
        """
        Orchestrates the generation of all 10 specific edge-case scenarios designed to 
        stress-test the completeness, optimality, and tie-breaking logic of the search algorithms.
        """
        cls.ensure_output_directory_exists()

        # Case 1: Standard Search Space
        # Represents the baseline multi-path graph provided in the assignment PDF.
        cls.generate_test_file(
            filename="T01_Standard.txt",
            nodes={1: (4, 1), 2: (2, 2), 3: (4, 4), 4: (6, 3), 5: (5, 6), 6: (7, 5)},
            adjacency_list={
                2: {1: 4.0, 3: 4.0},
                3: {1: 5.0, 2: 5.0, 5: 6.0, 6: 7.0},
                1: {3: 5.0, 4: 6.0},
                4: {1: 6.0, 3: 5.0, 5: 7.0},
                5: {3: 6.0, 4: 8.0},
                6: {3: 7.0},
            },
            origin=2,
            destinations=[5, 4],
        )

        # Case 2: Unreachable Goal (Disconnected Graph)
        # Tests whether the algorithms can cleanly exhaust the search space and return None 
        # without crashing or entering infinite loops when no path exists.
        cls.generate_test_file(
            filename="T02_Unreachable.txt",
            nodes={1: (0, 0), 2: (1, 1), 3: (10, 10)},
            adjacency_list={1: {2: 1.0}, 2: {1: 1.0}},  # Node 3 is physically isolated
            origin=1,
            destinations=[3],
        )

        # Case 3: Cycles and Infinite Loop Traps
        # Evaluates the closed-set (visited) cycle prevention logic. Node 1, 2, and 3 form 
        # an infinite loop, while the actual goal is offshoot Node 4.
        cls.generate_test_file(
            filename="T03_Cycles.txt",
            nodes={1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)},
            adjacency_list={
                1: {2: 1.0},
                2: {3: 1.0, 4: 5.0},
                3: {1: 1.0},
            },
            origin=1,
            destinations=[4],
        )

        # Case 4: Strict Tie-Breaking Enforcement
        # Node 1 connects to Node 2 and Node 3 with the exact SAME cost. Because 2 < 3, 
        # the strict assignment rule dictates Node 2 MUST be expanded before Node 3.
        cls.generate_test_file(
            filename="T04_TieBreak.txt",
            nodes={1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0)},
            adjacency_list={1: {2: 1.0, 3: 1.0}, 2: {4: 1.0}, 3: {4: 1.0}},
            origin=1,
            destinations=[4],
        )

        # Case 5: Linear Chain
        # A straight line topology with zero branching factor. Useful for verifying baseline 
        # O(N) complexity for all algorithms.
        cls.generate_test_file(
            filename="T05_Linear.txt",
            nodes={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0)},
            adjacency_list={1: {2: 1.0}, 2: {3: 1.0}, 3: {4: 1.0}, 4: {5: 1.0}},
            origin=1,
            destinations=[5],
        )

        # Case 6: Multiple Destinations (Close vs Far)
        # Tests the algorithm's ability to find the dynamically closest goal out of an array 
        # of valid targets, evaluating multi-goal heuristic competence.
        cls.generate_test_file(
            filename="T06_MultiDest.txt",
            nodes={1: (0, 0), 2: (0, 10), 3: (0, 20), 4: (5, 0), 5: (10, 0)},
            adjacency_list={1: {2: 10.0, 4: 1.0}, 2: {3: 10.0}, 4: {5: 1.0}},
            origin=1,
            destinations=[3, 5],
        )

        # Case 7: The Heuristic Trap (GBFS vs A*)
        # Designed specifically to trick Greedy Best-First Search. Node 3 appears spatially 
        # closer to the goal (Node 4) pulling GBFS in, but actually has a worse total path 
        # cost than going around, proving A*'s superiority.
        cls.generate_test_file(
            filename="T07_HeuristicTrap.txt",
            nodes={1: (0, 0), 2: (10, 0), 3: (0, 10), 4: (11, 0)},
            adjacency_list={1: {2: 10.0, 3: 2.0}, 2: {4: 10.0}, 3: {4: 2.0}},
            origin=1,
            destinations=[4],
        )

        # Case 8: Cost vs Hops (BFS vs UCS)
        # BFS assumes all edges have a cost of 1, so it will take the path with the fewest hops, 
        # even if the total weight is massive (50.0). Uniform Cost Search (CUS1) will properly 
        # navigate the longer, cheaper path.
        cls.generate_test_file(
            filename="T08_CostVsHops.txt",
            nodes={1: (0, 0), 2: (2, 2), 3: (5, 5), 4: (1, -1), 5: (2, -2), 6: (3, -1)},
            adjacency_list={1: {2: 50.0, 4: 1.0}, 2: {3: 50.0}, 4: {5: 1.0}, 5: {6: 1.0}, 6: {3: 1.0}},
            origin=1,
            destinations=[3],
        )

        # Case 9: Dead End Backtracking
        # Tests DFS and IDA*'s ability to hit a dead end (Node 2) and correctly backtrack up 
        # the tree to find the actual pathway to the goal.
        cls.generate_test_file(
            filename="T09_DeadEnd.txt",
            nodes={1: (0, 0), 2: (-5, 0), 3: (5, 0), 4: (10, 0)},
            adjacency_list={1: {2: 1.0, 3: 1.0}, 3: {4: 1.0}},
            origin=1,
            destinations=[4],
        )

        # Case 10: Zero Cost Edges
        # Ensures that mathematical logic (specifically inside A* and Uniform Cost Search) 
        # does not break, divide by zero, or incorrectly prune pathways when edge weights are 0.
        cls.generate_test_file(
            filename="T10_ZeroCost.txt",
            nodes={1: (0, 0), 2: (1, 0), 3: (2, 0)},
            adjacency_list={1: {2: 0.0}, 2: {3: 0.0}},
            origin=1,
            destinations=[3],
        )


if __name__ == "__main__":
    TestFactory.provision_all_test_cases()