import heapq
from collections import deque
from models import SearchState


class SearchEngine:
    def __init__(self, graph):
        self.graph = graph
        self.node_count = 0
        self.timestamp = 0

    def _create_node(self, node_id, parent, g, method):
        self.node_count += 1
        self.timestamp += 1
        h = self.graph.heuristic(node_id)
        return SearchState(node_id, parent, g, h, method, self.timestamp)

    def _reconstruct(self, state):
        path = []
        curr = state
        while curr:
            path.append(curr.node_id)
            curr = curr.parent
        return path[::-1]

    def solve(self, method: str):
        # Dispatcher
        m = method.lower()
        if m == "dfs":
            return self._dfs()
        if m == "bfs":
            return self._bfs()
        if m == "gbfs":
            return self._priority_search("gbfs")
        if m == "as":
            return self._priority_search("as")
        if m == "cus1":
            return self._priority_search("cus1")
        if m == "cus2":
            return self._ida_star()
        return None

    def _dfs(self):
        start = self._create_node(self.graph.origin, None, 0, "dfs")
        stack = [start]
        visited = set()  # Standard DFS typically avoids cycles in the current path, but for route finding on graphs we track visited.
        # Spec implies tree search logic but graph finding usually needs visited set to avoid infinite loops on cycles.
        # Given "Tree Based Search" title, strictly it might mean no visited set, but that crashes on 1<->2.
        # However, to be safe and standard for "finding a path", we use a visited set (Graph Search).
        # Most assignments allow Graph Search for efficiency unless explicitly "Tree Search" (explode exponential).
        # Let's check requirements. "Algorithms to implement... DFS".
        # I'll use a visited set to be safe. But wait, DFS for pathfinding usually keeps visited for *current path* to allow finding other paths?
        # No, for "finding *a* path", graph search (global visited) is standard and efficient.

        visited.add(self.graph.origin)  # Add start to visited immediately

        # Wait, if I add to visited immediately, I accept the start node.
        # In the loop:

        while stack:
            curr = stack.pop()

            # Check goal
            if curr.node_id in self.graph.destinations:
                return curr.node_id, self.node_count, self._reconstruct(curr)

            # Expand
            neighbors = self.graph.get_neighbors(curr.node_id)
            # Sort DESCENDING by ID so that when pushed to stack, they are popped in ASCENDING ID order
            neighbors.sort(key=lambda x: x[0], reverse=True)

            for nid, cost in neighbors:
                if nid not in visited:
                    visited.add(nid)
                    stack.append(self._create_node(nid, curr, curr.g + cost, "dfs"))
        return None

    def _bfs(self):
        start = self._create_node(self.graph.origin, None, 0, "bfs")
        queue = deque([start])
        visited = {self.graph.origin}

        while queue:
            curr = queue.popleft()
            if curr.node_id in self.graph.destinations:
                return curr.node_id, self.node_count, self._reconstruct(curr)

            neighbors = self.graph.get_neighbors(curr.node_id)
            # Sort ASCENDING by ID
            neighbors.sort(key=lambda x: x[0])

            for nid, cost in neighbors:
                if nid not in visited:
                    visited.add(nid)
                    queue.append(self._create_node(nid, curr, curr.g + cost, "bfs"))
        return None

    def _priority_search(self, method):
        start = self._create_node(self.graph.origin, None, 0, method)
        open_set = [start]  # Heap
        closed = set()

        while open_set:
            curr = heapq.heappop(open_set)

            # Goal check deferred to expansion time? Standard A* does goal check when popped.
            if curr.node_id in self.graph.destinations:
                return curr.node_id, self.node_count, self._reconstruct(curr)

            if curr.node_id in closed:
                continue
            closed.add(curr.node_id)

            for nid, cost in self.graph.get_neighbors(curr.node_id):
                if nid not in closed:
                    # Heap push handles sorting via SearchState.__lt__
                    # Note: We do not check for existence in open_set and update.
                    # We simpler add duplicates, and filter by closed set when popped.
                    # This is valid "lazy" Dijkstra/A*.
                    heapq.heappush(
                        open_set, self._create_node(nid, curr, curr.g + cost, method)
                    )
        return None

    def _ida_star(self):
        # Iterative Deepening A*
        start_h = self.graph.heuristic(self.graph.origin)
        threshold = start_h

        while True:
            # Create start node for this iteration?
            # Or reuse? If we count "nodes created", strict adherence implies we might create fresh tree.
            # But the prompt is "nodes created" total.
            # We will start fresh search.
            start_node = self._create_node(self.graph.origin, None, 0, "cus2")

            t = self._ida_recursive(start_node, threshold, [self.graph.origin])

            if isinstance(t, SearchState):
                return t.node_id, self.node_count, self._reconstruct(t)

            if t == float("inf"):
                return None

            threshold = t

    def _ida_recursive(self, current, threshold, path_ids):
        f = current.g + current.h
        if f > threshold:
            return f

        if current.node_id in self.graph.destinations:
            return current

        min_val = float("inf")
        neighbors = self.graph.get_neighbors(current.node_id)

        # Children generation
        children = []
        for nid, cost in neighbors:
            if nid not in path_ids:  # Cycle checking for current path
                child = self._create_node(nid, current, current.g + cost, "cus2")
                children.append(child)

        # Sort children: Ascending f (g+h), then IDs
        children.sort(key=lambda s: (s.g + s.h, s.node_id))

        for child in children:
            res = self._ida_recursive(child, threshold, path_ids + [child.node_id])
            if isinstance(res, SearchState):
                return res
            if res < min_val:
                min_val = res

        return min_val
