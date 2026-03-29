"""
Traffic Routing Graphical User Interface
========================================
Provides a threaded Tkinter interface for interacting with the TBRGS search engine,
dynamic parameter injection, and multi-lane route visualisation.
"""

import tkinter as tk
from tkinter import ttk
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from src.core.graph import Graph
from src.core.engine import SearchEngine
from src.utils.speed_time_converter import calculate_travel_time
from src.utils.logger_setup import get_colored_logger
from src.utils.config import APP_CONFIG

logger = get_colored_logger(__name__)


class TrafficRoutingApp:
    def __init__(self, root: tk.Tk, map_filepath: str):
        self.root = root
        self.root.title("TBRGS - Traffic-Based Route Guidance System")
        self.root.geometry("900x680")

        self.map_filepath = map_filepath
        self.graph = Graph()

        # Dynamic Parameters loaded from config.json
        self.k_paths_var = tk.IntVar(value=APP_CONFIG["TBRGS_Settings"]["top_k_paths"])
        self.speed_limit_var = tk.DoubleVar(
            value=APP_CONFIG["TBRGS_Settings"]["speed_limit_kmh"]
        )
        self.delay_var = tk.DoubleVar(
            value=APP_CONFIG["TBRGS_Settings"]["intersection_delay_seconds"]
        )
        self.fallback_flow_var = tk.DoubleVar(
            value=APP_CONFIG["ML_Settings"]["unmonitored_flow_assumption"]
        )

        self.use_ml_traffic = tk.BooleanVar(
            value=APP_CONFIG["GUI_Defaults"]["enable_ml_on_startup"]
        )
        self.selected_algorithm = tk.StringVar(value="Top-K Yen's")
        self.selected_model = tk.StringVar(
            value=APP_CONFIG["GUI_Defaults"]["default_model"]
        )
        self.is_searching = False

        self.canvas_width = 600
        self.canvas_height = 660
        self.padding = 10

        try:
            self.graph.load_from_file(self.map_filepath)
            self.node_list_int = sorted(list(self.graph.node_coordinates.keys()))
            self.scats_mapping = self._generate_node_mapping(self.node_list_int)
            self.display_names = list(self.scats_mapping.values())

            self.origin_var = tk.StringVar(value=self.display_names[0])
            self.dest_var = tk.StringVar(value=self.display_names[-1])
        except Exception as e:
            logger.critical(f"GUI Initialization Failed: {e}")
            self.root.destroy()
            return

        self._build_ui()
        self._draw_network()
        self._update_detail_label(
            event=type("Event", (object,), {"widget": self.origin_cb})()
        )

    def _generate_node_mapping(self, nodes: List[int]) -> Dict[int, str]:
        import pandas as pd
        import re
        import textwrap

        mapping = {}
        self.full_names = {}
        csv_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "raw"
            / "Traffic_Count_Locations_with_LONG_LAT.csv"
        )

        try:
            df = pd.read_csv(csv_path)
            id_col = next(
                (
                    c
                    for c in df.columns
                    if any(x in c.lower() for x in ["scats", "site", "id"])
                ),
                None,
            )
            desc_col = next(
                (
                    c
                    for c in df.columns
                    if any(x in c.lower() for x in ["desc", "name", "location"])
                ),
                None,
            )

            if id_col and desc_col:
                for _, row in df.iterrows():
                    match = re.search(r"\d+", str(row[id_col]))
                    if match:
                        scats_id = int(match.group())
                        if scats_id in nodes:
                            self.full_names[scats_id] = str(row[desc_col]).title()
        except Exception:
            pass

        for node in nodes:
            name = self.full_names.get(node, f"Intersection Site {node}")
            self.full_names[node] = name
            truncated = textwrap.shorten(name, width=29, placeholder="...")
            mapping[node] = f"{truncated} ({node})"

        return mapping

    def _get_id_from_name(self, name: str) -> int:
        for node_id, display_name in self.scats_mapping.items():
            if display_name == name:
                return node_id
        return self.node_list_int[0]

    def _update_detail_label(self, event=None):
        o_id = self._get_id_from_name(self.origin_var.get())
        d_id = self._get_id_from_name(self.dest_var.get())

        if (
            event
            and hasattr(event, "widget")
            and event.widget == getattr(self, "origin_cb", None)
        ):
            visited = {o_id}
            queue = [o_id]
            while queue:
                curr = queue.pop(0)
                for neighbor in self.graph.adjacency_list.get(curr, {}):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            reachable_names = [
                name
                for nid, name in self.scats_mapping.items()
                if nid in visited and nid != o_id
            ]
            reachable_names.sort()

            self.dest_cb["values"] = reachable_names
            if reachable_names and self.dest_var.get() not in reachable_names:
                self.dest_var.set(reachable_names[0])
                d_id = self._get_id_from_name(reachable_names[0])

        o_name = self.full_names.get(o_id, f"Unknown ID {o_id}")
        d_name = self.full_names.get(d_id, f"Unknown ID {d_id}")
        self.selection_detail.config(
            text=f"Selected Origin:\n{o_name}\n\nSelected Destination:\n{d_name}"
        )

    def _build_ui(self):
        self.canvas = tk.Canvas(
            self.root, width=self.canvas_width, height=self.canvas_height, bg="#1e1e1e"
        )
        self.canvas.pack(side=tk.LEFT, padx=(20, 10), pady=10, expand=True)

        control_frame = tk.Frame(self.root, width=280)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 20), pady=10)
        control_frame.pack_propagate(False)

        tk.Label(
            control_frame,
            text="TBRGS Controls",
            font=("Arial", 16, "bold"),
            fg="#333333",
        ).pack(pady=(2, 3))

        # Origin / Dest
        tk.Label(
            control_frame, text="Origin Intersection:", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W)
        self.origin_cb = ttk.Combobox(
            control_frame,
            textvariable=self.origin_var,
            values=self.display_names,
            state="readonly",
        )
        self.origin_cb.pack(fill=tk.X, pady=(2, 5))

        tk.Label(
            control_frame, text="Destination Intersection:", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W)
        self.dest_cb = ttk.Combobox(
            control_frame,
            textvariable=self.dest_var,
            values=self.display_names,
            state="readonly",
        )
        self.dest_cb.pack(fill=tk.X, pady=(2, 5))

        self.selection_detail = tk.Label(
            control_frame,
            text="",
            fg="#666",
            justify=tk.LEFT,
            wraplength=300,
            font=("Arial", 8, "italic"),
        )
        self.selection_detail.pack(fill=tk.X, pady=(0, 0), anchor=tk.W)

        self.origin_cb.bind("<<ComboboxSelected>>", self._update_detail_label)
        self.dest_cb.bind("<<ComboboxSelected>>", self._update_detail_label)

        # Compact Algorithm & Model Selectors
        algo_frame = tk.Frame(control_frame)
        algo_frame.pack(fill=tk.X, pady=2)
        tk.Label(algo_frame, text="Algorithm:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT
        )
        self.algo_cb = ttk.Combobox(
            algo_frame,
            textvariable=self.selected_algorithm,
            values=[
                "Top-K Yen's",
                "A* Search (AS)",
                "Greedy Best-First",
                "Uniform Cost",
                "BFS",
                "DFS",
                "IDA*",
            ],
            state="readonly",
            width=15,
        )
        self.algo_cb.pack(side=tk.RIGHT)

        model_frame = tk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=2)
        tk.Label(model_frame, text="ML Model:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT
        )
        self.model_cb = ttk.Combobox(
            model_frame,
            textvariable=self.selected_model,
            values=APP_CONFIG["ML_Settings"]["supported_models"],
            state="readonly",
            width=15,
        )
        self.model_cb.pack(side=tk.RIGHT)

        # Dynamic Parameter Settings Panel
        param_frame = tk.LabelFrame(
            control_frame,
            text="Parameter Settings",
            font=("Arial", 10, "bold"),
            fg="#333",
        )
        param_frame.pack(fill=tk.X, pady=5)

        def add_param(parent, label_text, var, row):
            tk.Label(parent, text=label_text, font=("Arial", 9)).grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=2
            )
            tk.Entry(parent, textvariable=var, width=6, justify="center").grid(
                row=row, column=1, sticky=tk.E, padx=5, pady=2
            )

        add_param(param_frame, "Number of Paths (K):", self.k_paths_var, 0)
        add_param(param_frame, "Speed Limit (km/h):", self.speed_limit_var, 1)
        add_param(param_frame, "Intersection Delay (s):", self.delay_var, 2)
        add_param(param_frame, "Unmonitored Flow (v/h):", self.fallback_flow_var, 3)
        param_frame.columnconfigure(0, weight=1)

        tk.Checkbutton(
            control_frame,
            text="Enable ML Traffic Validation",
            variable=self.use_ml_traffic,
            font=("Arial", 10, "bold"),
            fg="#4CAF50",
        ).pack(anchor=tk.W, pady=2)

        self.search_btn = tk.Button(
            control_frame,
            text="Calculate Route",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            command=self.start_threaded_search,
        )
        self.search_btn.pack(pady=5, fill=tk.X)

        self.status_label = tk.Label(
            control_frame, text="System Ready.", fg="gray", justify=tk.LEFT
        )
        self.status_label.pack(anchor=tk.W)

    def _apply_ml_weights(self, target_graph: Graph, model_type: str) -> None:
        import re

        target_graph.is_time_based = True

        base_db_path = (
            Path(__file__).resolve().parent.parent.parent / "data" / "database"
        )
        db_path_model = base_db_path / f"traffic_state_{model_type}.db"
        db_path_general = base_db_path / "traffic_state.db"

        active_db = db_path_model if db_path_model.exists() else db_path_general
        if not active_db.exists():
            return

        predictions = {}
        with sqlite3.connect(active_db) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT scats_id, predicted_flow FROM predictions")
                for row in cursor.fetchall():
                    match = re.search(r"\d+", str(row[0]))
                    if match:
                        predictions[int(match.group())] = float(row[1])
            except sqlite3.OperationalError:
                return

        delay_hrs = self.delay_var.get() / 3600.0
        fallback_flow = self.fallback_flow_var.get()

        for src, neighbors in target_graph.adjacency_list.items():
            for dst, physical_dist in neighbors.items():
                flow = predictions.get(dst, fallback_flow)
                target_graph.adjacency_list[src][dst] = (
                    calculate_travel_time(flow, physical_dist) + delay_hrs
                )

    def start_threaded_search(self) -> None:
        if self.is_searching:
            return
        self.is_searching = True
        self.search_btn.config(state=tk.DISABLED, text="Executing...", bg="#888")
        self.status_label.config(text="Status: Computing optimal path...", fg="orange")
        threading.Thread(target=self._execute_search_task, daemon=True).start()

    def _execute_search_task(self) -> None:
        o_id = self._get_id_from_name(self.origin_var.get())
        d_id = self._get_id_from_name(self.dest_var.get())

        try:
            temp_graph = Graph()
            temp_graph.load_from_file(self.map_filepath)
            temp_graph.origin = o_id
            temp_graph.destinations = [d_id]

            # Base Kinematics loaded from UI Parameters
            temp_graph.is_time_based = True
            base_speed = self.speed_limit_var.get()
            delay_hrs = self.delay_var.get() / 3600.0

            for src, neighbors in temp_graph.adjacency_list.items():
                for dst, physical_dist in neighbors.items():
                    temp_graph.adjacency_list[src][dst] = (
                        physical_dist / base_speed
                    ) + delay_hrs

            if self.use_ml_traffic.get():
                self._apply_ml_weights(temp_graph, self.selected_model.get())

            # Parse requested algorithm
            algo_map = {
                "Top-K Yen's": f"topk-{self.k_paths_var.get()}",
                "A* Search (AS)": "as",
                "Greedy Best-First": "gbfs",
                "Uniform Cost": "cus1",
                "BFS": "bfs",
                "DFS": "dfs",
                "IDA*": "cus2",
            }
            target_method = algo_map.get(self.selected_algorithm.get(), "as")

            engine = SearchEngine(temp_graph)
            result = engine.solve(target_method)
            self.root.after(0, self._finalize_ui, result, o_id, d_id, target_method)

        except Exception as e:
            logger.error(f"Execution Error: {e}")
            self.root.after(0, self._finalize_ui, None, o_id, d_id, "")

    def _finalize_ui(self, result: Any, origin: int, dest: int, method: str) -> None:
        self.is_searching = False
        self.search_btn.config(state=tk.NORMAL, text="Calculate Route", bg="#4CAF50")

        if result:
            if method.startswith("topk") and isinstance(result, list):
                status_text = f"Success: {len(result)} Alternative Paths Found.\n"
                for i, (goal, nodes, path) in enumerate(result):
                    status_text += (
                        f"Path {i+1}: {len(path)} steps (Nodes Explored: {nodes})\n"
                    )
                self.status_label.config(text=status_text, fg="#4CAF50")
                all_paths = [p[2] for p in result]
                self._draw_network(
                    active_paths=all_paths, current_origin=origin, current_dest=dest
                )
            else:
                goal, nodes, path = result
                self.status_label.config(
                    text=f"Success: Route Optimal.\nGoal Reached: {goal}\nNodes Explored: {nodes}\nSteps: {len(path)}",
                    fg="#4CAF50",
                )
                self._draw_network(
                    active_paths=[path], current_origin=origin, current_dest=dest
                )
        else:
            self.status_label.config(text="Exception: No valid path exists.", fg="red")
            self._draw_network(current_origin=origin, current_dest=dest)

    def _transform_coordinates(self):
        lons = [c[0] for c in self.graph.node_coordinates.values()]
        lats = [c[1] for c in self.graph.node_coordinates.values()]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        def scale(lon, lat):
            x_n = (lon - min_lon) / (max_lon - min_lon) if max_lon != min_lon else 0.5
            y_n = (lat - min_lat) / (max_lat - min_lat) if max_lat != min_lat else 0.5
            return (
                self.padding + x_n * (self.canvas_width - 2 * self.padding),
                self.canvas_height
                - (self.padding + y_n * (self.canvas_height - 2 * self.padding)),
            )

        return scale

    def _draw_network(
        self, active_paths=None, current_origin=None, current_dest=None
    ) -> None:
        self.canvas.delete("all")
        scale = self._transform_coordinates()

        for src, neighbors in self.graph.adjacency_list.items():
            x1, y1 = scale(*self.graph.node_coordinates[src])
            for dst in neighbors:
                x2, y2 = scale(*self.graph.node_coordinates[dst])
                self.canvas.create_line(
                    x1, y1, x2, y2, fill="#E0E0E0", width=2, dash=(3, 3)
                )

        if active_paths:
            colors = ["#00FF00", "#FFD700", "#FF8C00", "#9370DB", "#1E90FF"]
            widths = [5, 4, 3, 2, 2]

            for i in reversed(range(len(active_paths))):
                path = active_paths[i]
                c = colors[i % len(colors)]
                w = widths[i % len(widths)]

                # THE MULTI-LANE VISUALIZER FIX: Fanning out identical roads
                # This shifts identical paths by a few pixels so you can see all 5 colors side-by-side
                offset = i * 2.5

                for j in range(len(path) - 1):
                    x1, y1 = scale(*self.graph.node_coordinates[path[j]])
                    x2, y2 = scale(*self.graph.node_coordinates[path[j + 1]])
                    self.canvas.create_line(
                        x1 + offset,
                        y1 - offset,
                        x2 + offset,
                        y2 - offset,
                        fill=c,
                        width=w,
                    )

        o = (
            current_origin
            if current_origin is not None
            else self._get_id_from_name(self.origin_var.get())
        )
        d = (
            current_dest
            if current_dest is not None
            else self._get_id_from_name(self.dest_var.get())
        )

        for nid, coords in self.graph.node_coordinates.items():
            x, y = scale(*coords)
            color, sz = ("#FFFFFF", 2.6)
            if nid == o:
                color, sz = ("#00BFFF", 8)
            elif nid == d:
                color, sz = ("#FF4500", 8)
            self.canvas.create_oval(
                x - sz, y - sz, x + sz, y + sz, fill=color, outline=color
            )


def launch_gui(map_filepath: str):
    root = tk.Tk()
    TrafficRoutingApp(root, map_filepath)
    root.mainloop()
