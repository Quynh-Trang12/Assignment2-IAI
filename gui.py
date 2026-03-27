import tkinter as tk
from tkinter import ttk, messagebox
import threading

from graph import Graph
from predictor import FlowPredictor
from sequence_store import SequenceStore
from search import SearchCLI
from k_shortest import solve_k_paths

class RouteFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TBRGS - Traffic Based Route Guidance System")
        self.root.geometry("750x600")
        
        self.ts_df = None
        self.store = None
        self.graph = None
        self.predictor = None
        
        self._build_ui()
        
        # Load data in background to prevent freezing
        self.status_var.set("Loading dataset and initializing sequence store...")
        threading.Thread(target=self._initialize_backend, daemon=True).start()

    def _build_ui(self):
        # Top Frame for Inputs
        input_frame = ttk.LabelFrame(self.root, text="Search Parameters", padding=(10, 10))
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.node_options = [
            "Node 1 (SCATS 0970)",
            "Node 2 (SCATS 2000)",
            "Node 3 (SCATS 3002)",
            "Node 4 (SCATS 2827)",
            "Node 5 (SCATS 3126)",
            "Node 6 (SCATS 4335)"
        ]
        
        ttk.Label(input_frame, text="Origin Node:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.origin_var = tk.StringVar(value="Node 2 (SCATS 2000)")
        self.origin_entry = ttk.Combobox(input_frame, textvariable=self.origin_var, values=self.node_options, state="readonly", width=20)
        self.origin_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(input_frame, text="Destination Node:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(15, 0))
        self.dest_var = tk.StringVar(value="Node 5 (SCATS 3126)")
        self.dest_entry = ttk.Combobox(input_frame, textvariable=self.dest_var, values=self.node_options, state="readonly", width=20)
        self.dest_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(input_frame, text="Predictive Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="rf")
        self.model_combo = ttk.Combobox(input_frame, textvariable=self.model_var, 
                                        values=["lstm", "gru", "rf", "baseline", "mock"], state="readonly", width=12)
        self.model_combo.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(input_frame, text="Time Index (t):").grid(row=1, column=2, sticky=tk.W, pady=5, padx=(15, 0))
        self.time_var = tk.IntVar(value=100)
        self.time_entry = ttk.Entry(input_frame, textvariable=self.time_var, width=15)
        self.time_entry.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(input_frame, text="Top K Routes:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.k_var = tk.IntVar(value=5)
        self.k_entry = ttk.Entry(input_frame, textvariable=self.k_var, width=15)
        self.k_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        self.search_btn = ttk.Button(input_frame, text="Find Optimal Routes", command=self._on_search_clicked, state=tk.DISABLED)
        self.search_btn.grid(row=2, column=3, sticky=tk.E, pady=10)

        # Status Bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Output Results Area
        res_frame = ttk.LabelFrame(self.root, text="Recommended Routes (K-Shortest)", padding=(10, 10))
        res_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Scrollable textual output
        self.output_text = tk.Text(res_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        scroll = ttk.Scrollbar(res_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scroll.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _initialize_backend(self):
        try:
            self.ts_df = SearchCLI._load_ts_dataframe()
            self.store = SequenceStore(self.ts_df)
            
            # Use mock predictor initially just to seed the engine
            self.predictor = FlowPredictor("mock", self.store)
            self.graph = Graph(self.predictor)
            self.graph.debug = True
            
            # Hardcoded test map as default fallback if we don't have complete map geometry
            self.graph.load_from_file("PathFinder-test.txt")
            
            # Enable the UI
            self.root.after(0, lambda: self.search_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("System Ready. Waiting for search query..."))
            
            # Map default edges
            # Normally we would use a full Borondara map here if provided, 
            # but we use the existing assignment test text format as the graph geometry.
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Initialization Error: {e}"))

    def _on_search_clicked(self):
        self.search_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Calculating Top-{self.k_var.get()} paths using {self.model_var.get().upper()}...")
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Executing pathfinding algorithms...\n\n")
        self.output_text.config(state=tk.DISABLED)
        
        threading.Thread(target=self._run_search, daemon=True).start()
        
    def _run_search(self):
        try:
            # Parse integer ID from "Node 2 (SCATS 2000)"
            origin = int(self.origin_var.get().split()[1])
            dest = int(self.dest_var.get().split()[1])
            k = int(self.k_var.get())
            t = int(self.time_var.get())
            model_type = self.model_var.get()
            
            # Re-initialize predictor with correct model
            self.predictor = FlowPredictor(model_type, self.store)
            self.graph.predictor = self.predictor
            
            # Calculate path
            paths = solve_k_paths(self.graph, origin, dest, k_paths=k, time_context={"t": t})
            
            # Format output
            self.root.after(0, self._render_results, paths, origin, dest, model_type, t)
            
        except ValueError:
            self.root.after(0, lambda: messagebox.showerror("Input Error", "Node IDs must be valid selections and Time Index must be an integer."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Execution Error", str(e)))
        finally:
            self.root.after(0, lambda: self.search_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Execution Complete."))
            
    def _render_results(self, paths, origin, dest, model_type, time_idx):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        
        if not paths:
            self.output_text.insert(tk.END, f"No valid routes found from {origin} to {dest}.\n")
            self.output_text.config(state=tk.DISABLED)
            return
            
        header = f"Top-{len(paths)} Optimal Routes\n"
        header += f"Config: Origin={origin}, Dest={dest}, Model={model_type.upper()}, t={time_idx}\n"
        header += "="*60 + "\n\n"
        self.output_text.insert(tk.END, header)
        
        for idx, route in enumerate(paths, start=1):
            cost = route["cost"]
            sequence = " \u2192 ".join(map(str, route["path"]))
            msg = f"Route #{idx}:  {cost:.2f} mins\n   Path: {sequence}\n\n"
            self.output_text.insert(tk.END, msg)
            
        self.output_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = RouteFinderGUI(root)
    root.mainloop()
