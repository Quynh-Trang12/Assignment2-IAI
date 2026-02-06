import os
from typing import List, Dict


class TestFactory:
    """
    Generates the 10 specific test cases required for Assignment 2A robustness testing.
    Writes strictly formatted .txt files to the defined output directory.
    """

    OUTPUT_DIR = "tests/cases"

    @staticmethod
    def ensure_dir():
        if not os.path.exists(TestFactory.OUTPUT_DIR):
            os.makedirs(TestFactory.OUTPUT_DIR)

    @staticmethod
    def write_case(
        filename: str, nodes: Dict, edges: Dict, origin: int, dests: List[int]
    ):
        filepath = os.path.join(TestFactory.OUTPUT_DIR, filename)
        # Ensure path uses OS specific separator if needed, but relative path is fine
        with open(filepath, "w") as f:
            f.write("Nodes:\n")
            for nid, coords in nodes.items():
                f.write(f"{nid}: ({coords[0]},{coords[1]})\n")

            f.write("Edges:\n")
            for u, neighbors in edges.items():
                for v, cost in neighbors.items():
                    f.write(f"({u},{v}): {cost}\n")

            f.write("Origin:\n")
            f.write(f"{origin}\n")

            f.write("Destinations:\n")
            # Strict formatting: semi-colon separated
            f.write(f"{'; '.join(map(str, dests))}\n")
        print(f"[Factory] Generated: {filepath}")

    @classmethod
    def generate_all(cls):
        cls.ensure_dir()

        # Case 1: Standard (From PDF)
        cls.write_case(
            "T01_Standard.txt",
            nodes={1: (4, 1), 2: (2, 2), 3: (4, 4), 4: (6, 3), 5: (5, 6), 6: (7, 5)},
            edges={
                2: {1: 4, 3: 4},
                3: {1: 5, 2: 5, 5: 6, 6: 7},
                1: {3: 5, 4: 6},
                4: {1: 6, 3: 5, 5: 7},
                5: {3: 6, 4: 8},
                6: {3: 7},
            },
            origin=2,
            dests=[5, 4],
        )

        # Case 2: Unreachable Goal (Disconnected)
        cls.write_case(
            "T02_Unreachable.txt",
            nodes={1: (0, 0), 2: (1, 1), 3: (10, 10)},
            edges={1: {2: 1}, 2: {1: 1}},  # 3 is isolated
            origin=1,
            dests=[3],
        )

        # Case 3: Cycles (Infinite Loop Trap)
        cls.write_case(
            "T03_Cycles.txt",
            nodes={1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)},
            edges={
                1: {2: 1},
                2: {3: 1, 4: 5},
                3: {1: 1},
            },  # 1-2-3-1 Loop. 4 is goal.
            origin=1,
            dests=[4],
        )

        # Case 4: Tie Breaking (Ascending ID check)
        # Node 1 connects to 2 and 3 with SAME cost. 2 < 3, so 2 should be visited first.
        cls.write_case(
            "T04_TieBreak.txt",
            nodes={1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0)},
            edges={1: {2: 1, 3: 1}, 2: {4: 1}, 3: {4: 1}},
            origin=1,
            dests=[4],
        )

        # Case 5: Linear Line
        cls.write_case(
            "T05_Linear.txt",
            nodes={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0)},
            edges={1: {2: 1}, 2: {3: 1}, 3: {4: 1}, 4: {5: 1}},
            origin=1,
            dests=[5],
        )

        # Case 6: Multiple Destinations (Close vs Far)
        # 1->2 (cost 10) -> Goal 3
        # 1->4 (cost 1) -> Goal 5
        cls.write_case(
            "T06_MultiDest.txt",
            nodes={1: (0, 0), 2: (0, 10), 3: (0, 20), 4: (5, 0), 5: (10, 0)},
            edges={1: {2: 10, 4: 1}, 2: {3: 10}, 4: {5: 1}},
            origin=1,
            dests=[3, 5],
        )

        # Case 7: Heuristic vs Cost (Greedy trap)
        cls.write_case(
            "T07_HeuristicTrap.txt",
            nodes={1: (0, 0), 2: (10, 0), 3: (0, 10), 4: (11, 0)},
            edges={1: {2: 10, 3: 2}, 2: {4: 10}, 3: {4: 2}},
            origin=1,
            dests=[4],
        )

        # Case 8: Cost vs Hops (BFS vs UCS)
        cls.write_case(
            "T08_CostVsHops.txt",
            nodes={1: (0, 0), 2: (2, 2), 3: (5, 5), 4: (1, -1), 5: (2, -2), 6: (3, -1)},
            edges={1: {2: 50, 4: 1}, 2: {3: 50}, 4: {5: 1}, 5: {6: 1}, 6: {3: 1}},
            origin=1,
            dests=[3],
        )

        # Case 9: Dead End Backtracking
        cls.write_case(
            "T09_DeadEnd.txt",
            nodes={1: (0, 0), 2: (-5, 0), 3: (5, 0), 4: (10, 0)},
            edges={1: {2: 1, 3: 1}, 3: {4: 1}},
            origin=1,
            dests=[4],
        )

        # Case 10: Zero Cost Edges (Valid, if unusual)
        cls.write_case(
            "T10_ZeroCost.txt",
            nodes={1: (0, 0), 2: (1, 0), 3: (2, 0)},
            edges={1: {2: 0}, 2: {3: 0}},
            origin=1,
            dests=[3],
        )


if __name__ == "__main__":
    TestFactory.generate_all()
