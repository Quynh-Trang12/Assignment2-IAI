import math
from src.utils.logger_setup import get_colored_logger
from src.utils.config import APP_CONFIG

# Instantiate the centralized, color-coded logger
logger = get_colored_logger(__name__)

peak_flow = APP_CONFIG["ML_Settings"]["capacity_threshold_veh_hr"]
speed_limit = APP_CONFIG["TBRGS_Settings"]["speed_limit_kmh"]
under_capacity_flow = APP_CONFIG["ML_Settings"]["flow_at_under_capacity"]


def calculate_expected_speed(predicted_flow: float) -> float:
    """
    Calculates the expected vehicle speed (km/h) based on predicted traffic flow.
    Adhere to the mathematical constraints and assumptions, as specified in Assignment 2B.
    """
    flow = max(0.0, float(predicted_flow))

    # Rule 1: Speed Limit Cap
    if flow <= under_capacity_flow:
        return speed_limit

    # Rule 2: Parabola Turning Point Capacity
    is_over_capacity = False
    if flow >= peak_flow:
        flow = peak_flow
        is_over_capacity = True

    # Rule 3: The Quadratic Formula
    # Equation: flow = -1.4648375*(speed)^2 + 93.75*(speed)
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = (b**2) - (4 * a * c)

    # Float precision safeguard (prevents -0.0000000001 from crashing math.sqrt)
    if discriminant < 0:
        discriminant = 0.0

    # Calculate both mathematical roots
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)

    # Rule 4: Red vs. Green Curve Selection (Congestion vs Under-capacity traffic conditions)
    if not is_over_capacity:
        # Green Line: Under capacity uses the higher speed root
        expected_speed = max(root1, root2)
    else:
        # Red Line / Vertex: At or over capacity uses the lower speed root
        expected_speed = min(root1, root2)

    return expected_speed


def calculate_travel_time(flow: float, physical_distance: float) -> float:
    """
    Converts physical distance to travel time (hours) using
    quadratic flow-speed converter.
    """
    # Strictly use the quadratic conversion defined by the PDF
    actual_speed = calculate_expected_speed(flow)

    # Time = Distance / Speed
    return physical_distance / actual_speed


# ---------------------------------------------------------
# Unit Testing / Standalone Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    test_flows = [100, 350, 351, 800, 1200, 1500, 2000]
    distance = 2.5  # Example: 2.5 kilometers

    logger.info("Executing Kinematics Validation Tests (Distance: 2.5km)...")
    for f in test_flows:
        spd = calculate_expected_speed(f)
        tm = calculate_travel_time(f, distance)
        # Convert hours to minutes for human-readable terminal output
        logger.info(
            f"Flow: {f:4} veh/hr -> Speed: {spd:5.2f} km/h | Travel Time: {tm * 60:5.2f} minutes"
        )
