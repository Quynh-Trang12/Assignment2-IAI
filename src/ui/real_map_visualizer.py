"""
TBRGS Real-World Interactive Map Visualizer
===========================================
Fulfills the Research Initiative requirement by projecting the mathematical
SCATS topology onto an interactive, real-world OpenStreetMap layer using Folium.
"""

import sys
from pathlib import Path

# Path Resolution
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import folium
    from folium.plugins import MarkerCluster
    from src.core.graph import Graph
    from src.utils.logger_setup import get_colored_logger
except ImportError as e:
    print(f"FAILED TO IMPORT REQUIRED LIBRARIES. Run: pip install folium\nError: {e}")
    sys.exit(1)

logger = get_colored_logger(__name__)


def generate_interactive_map():
    map_path = _PROJECT_ROOT / "data" / "maps" / "map.txt"
    output_html = _PROJECT_ROOT / "data" / "maps" / "boroondara_interactive.html"

    logger.info("Initializing Real-World Interactive Map Generator...")

    g = Graph()
    try:
        g.load_from_file(str(map_path))
        logger.info(
            f"Loaded {len(g.node_coordinates)} SCATS nodes from mathematical topology."
        )
    except Exception as e:
        logger.error(f"Failed to load map.txt: {e}")
        return

    # 1. Initialize the Map with Advanced UI/UX Settings
    m = folium.Map(
        location=[-37.825, 145.050],
        zoom_start=12,
        zoom_control=False,  # Removes the +/- buttons
        tiles="OpenStreetMap",
        zoom_snap=0.25,  # Enables smooth fractional zooming
        zoom_delta=0.25,  # Slows down the zoom step size
        wheel_px_per_zoom_level=120,  # Smoothes out the scroll wheel feel
        max_bounds=True,  # Locks panning to the limits below
        min_lat=-38.120,
        max_lat=-37.520,
        min_lon=144.950,
        max_lon=145.150,
    )

    # 2. Draw the Boroondara Boundary Polygon
    # Folium uses (Latitude, Longitude) format for plotting
    boroondara_polygon = [
        (-37.815, 144.985),  # West edge (Hawthorn / Yarra River)
        (-37.765, 145.010),  # NW edge (Kew / Yarra Bend)
        (-37.755, 145.050),  # Mid-North edge (Kew East / River Bend)
        (-37.775, 145.090),  # NE edge (Balwyn North / Eastern Freeway)
        (-37.830, 145.120),  # East edge (Surrey Hills / Warrigal Rd)
        (-37.880, 145.100),  # SE edge (Ashburton)
        (-37.875, 145.040),  # South edge (Glen Iris / Gardiners Creek)
        (-37.845, 145.020),  # SW edge (Kooyong)
        (-37.835, 144.990),  # SW edge (Hawthorn)
    ]

    folium.Polygon(
        locations=boroondara_polygon,
        color="#0052cc",
        weight=2,
        fill=True,
        fill_color="#0052cc",
        fill_opacity=0.1,  # Highly transparent so street names remain crystal clear
    ).add_to(m)

    # 3. Plot the Validated Road Corridors FIRST (so they sit below the markers)
    drawn_edges = set()
    for src, neighbors in g.adjacency_list.items():
        for dst in neighbors:
            edge_tuple = tuple(sorted((src, dst)))
            if edge_tuple not in drawn_edges:
                src_lat, src_lon = (
                    g.node_coordinates[src][1],
                    g.node_coordinates[src][0],
                )
                dst_lat, dst_lon = (
                    g.node_coordinates[dst][1],
                    g.node_coordinates[dst][0],
                )

                folium.PolyLine(
                    locations=[(src_lat, src_lon), (dst_lat, dst_lon)],
                    color="#888888",
                    weight=3,
                    opacity=0.5,  # Slightly higher opacity to contrast with the blue background
                    dash_array="5, 5",
                ).add_to(m)
                drawn_edges.add(edge_tuple)

    # 4. Initialize MarkerCluster for massive node grouping
    # This automatically hides overlapping dots and shows a numbered group until you zoom in
    marker_cluster = MarkerCluster(
        name="SCATS Intersections",
        overlay=True,
        control=True,
        icon_create_function=None,
    ).add_to(m)

    # 5. Plot the SCATS Intersections into the Cluster
    for node_id, (lon, lat) in g.node_coordinates.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            popup=f"<strong>SCATS Site ID:</strong> {node_id}",
            tooltip=f"Site {node_id}",
            color="#d32f2f",  # Switched to Red to pop against the blue boundary
            weight=1,
            fill=True,
            fill_color="#d32f2f",
            fill_opacity=0.9,
        ).add_to(
            marker_cluster
        )  # Add to the cluster instead of the base map

    # 6. Export to Interactive HTML
    m.save(str(output_html))
    logger.info("Interactive Map successfully generated!")
    logger.info(f"▶ OPEN THIS FILE IN YOUR BROWSER: {output_html}")


if __name__ == "__main__":
    generate_interactive_map()
