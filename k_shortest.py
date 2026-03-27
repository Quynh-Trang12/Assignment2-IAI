import heapq
from typing import List, Dict, Any, Optional, Tuple
from graph import Graph

def solve_k_paths(graph: Graph, origin: int, destination: int, k_paths: int = 5, time_context: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Find the K-shortest paths using Yen's algorithm.
    Returns:
        List of dicts containing {"path": [nodes...], "cost": float}
    """
    def _shortest_path(start: int, dest: int, mask_edges: set, mask_nodes: set) -> Tuple[float, List[int]]:
        open_set = [(0.0, start, [start])]
        closed_set = set()
        
        while open_set:
            cost, current, path = heapq.heappop(open_set)
            
            if current == dest:
                return cost, path
                
            if current in closed_set:
                continue
            closed_set.add(current)
            
            for neighbor in graph.get_neighbors(current):
                if neighbor in closed_set or neighbor in mask_nodes:
                    continue
                if (current, neighbor) in mask_edges:
                    continue
                    
                edge_time = graph.get_edge_cost(current, neighbor, time_context)
                new_cost = cost + edge_time
                heapq.heappush(open_set, (new_cost, neighbor, path + [neighbor]))
                
        return float('inf'), []

    A = []
    B = []
    
    cost_0, path_0 = _shortest_path(origin, destination, set(), set())
    if not path_0:
        return A
        
    A.append({"path": path_0, "cost": cost_0})
    
    for k in range(1, k_paths):
        for i in range(len(A[k-1]["path"]) - 1):
            spur_node = A[k-1]["path"][i]
            root_path = A[k-1]["path"][:i+1]
            
            mask_edges = set()
            for p_dict in A:
                p = p_dict["path"]
                if len(p) > i and p[:i+1] == root_path:
                    mask_edges.add((p[i], p[i+1]))
                    
            mask_nodes = set(root_path[:-1])
            
            spur_cost, spur_path = _shortest_path(spur_node, destination, mask_edges, mask_nodes)
            
            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_cost = 0.0
                for idx in range(len(total_path)-1):
                    total_cost += graph.get_edge_cost(total_path[idx], total_path[idx+1], time_context)
                    
                path_dict = {"path": total_path, "cost": total_cost}
                
                # Verify it does not already exist
                if not any(b["path"] == path_dict["path"] for b in B):
                    B.append(path_dict)
                    
        if not B:
            break
            
        B.sort(key=lambda x: x["cost"])
        A.append(B.pop(0))
        
    return A
