# ---------------------------------------------------------------------------
# travel_time.py
# ---------------------------------------------------------------------------
# Converts predicted traffic flow into speed, then converts speed + distance
# into travel time for Assignment 2B.
#
# Assumptions from the provided assignment materials:
# - Speed limit is capped at 60 km/h
# - Average controlled-intersection delay is 30 seconds
# - Flow-speed relationship:
#       flow = -1.4648375 * speed^2 + 93.75 * speed
# ---------------------------------------------------------------------------

import math


# Constants from the assignment spec / conversion document
A = -1.4648375
B = 93.75
C = 0.0

SPEED_LIMIT_KMH = 60.0
DEFAULT_INTERSECTION_DELAY_SEC = 30.0


def flow_to_speed(flow: float, use_under_capacity_branch: bool = True) -> float:
    """
    Convert traffic flow (vehicles/hour) into speed (km/h) using the quadratic:

        flow = -1.4648375 * speed^2 + 93.75 * speed

    Rearranged into standard quadratic form:
        -1.4648375 * speed^2 + 93.75 * speed - flow = 0

    Args:
        flow: Predicted traffic flow in vehicles per hour.
        use_under_capacity_branch:
            True  -> use the larger root (green branch / under-capacity traffic)
            False -> use the smaller root (red branch / over-capacity traffic)

    Returns:
        Speed in km/h, capped at 60 km/h and never below 0.

    Notes:
        - For Assignment 2B, the provided document says you may assume the traffic
          at each segment is under capacity, so the default is True.
        - If the flow is small enough that the computed speed exceeds 60 km/h,
          the result is capped at 60.
    """
    if flow <= 0:
        return SPEED_LIMIT_KMH

    # Quadratic coefficients for:
    # A*s^2 + B*s - flow = 0
    a = A
    b = B
    c = -flow

    discriminant = b * b - 4 * a * c

    # Guard against numerical or invalid cases
    if discriminant < 0:
        return SPEED_LIMIT_KMH

    sqrt_discriminant = math.sqrt(discriminant)

    root1 = (-b + sqrt_discriminant) / (2 * a)
    root2 = (-b - sqrt_discriminant) / (2 * a)

    # Keep only physically meaningful speeds
    valid_roots = [r for r in (root1, root2) if r >= 0]

    if not valid_roots:
        return SPEED_LIMIT_KMH

    # Under-capacity branch = larger speed
    # Over-capacity branch = smaller speed
    speed = max(valid_roots) if use_under_capacity_branch else min(valid_roots)

    # Cap at speed limit
    return min(speed, SPEED_LIMIT_KMH)


def compute_travel_time_minutes(
    distance_km: float,
    speed_kmh: float,
    intersection_delay_sec: float = DEFAULT_INTERSECTION_DELAY_SEC,
) -> float:
    """
    Compute travel time in minutes for one edge.

    Args:
        distance_km: Distance of the road segment in km.
        speed_kmh: Speed on that segment in km/h.
        intersection_delay_sec: Extra delay in seconds for the controlled intersection.

    Returns:
        Travel time in minutes.
    """
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")

    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be greater than 0")

    travel_hours = distance_km / speed_kmh
    travel_minutes = travel_hours * 60.0
    delay_minutes = intersection_delay_sec / 60.0

    return travel_minutes + delay_minutes


def flow_to_travel_time_minutes(
    flow: float,
    distance_km: float,
    intersection_delay_sec: float = DEFAULT_INTERSECTION_DELAY_SEC,
    use_under_capacity_branch: bool = True,
) -> float:
    """
    Full pipeline:
        flow -> speed -> travel time

    Args:
        flow: Traffic flow in vehicles/hour.
        distance_km: Segment distance in km.
        intersection_delay_sec: Added delay at intersection in seconds.
        use_under_capacity_branch: Which root of the quadratic to use.

    Returns:
        Travel time in minutes.
    """
    speed_kmh = flow_to_speed(flow, use_under_capacity_branch=use_under_capacity_branch)
    return compute_travel_time_minutes(
        distance_km=distance_km,
        speed_kmh=speed_kmh,
        intersection_delay_sec=intersection_delay_sec,
    )


if __name__ == "__main__":
    # Quick sanity checks
    sample_flows = [0, 100, 351, 400, 800, 1200]
    sample_distance = 2.0  # km

    print("Flow -> Speed -> Travel Time")
    for flow in sample_flows:
        speed = flow_to_speed(flow)
        time_min = flow_to_travel_time_minutes(flow, sample_distance)
        print(
            f"flow={flow:>5.1f} veh/hr | speed={speed:>6.2f} km/h | "
            f"time={time_min:>6.2f} min for {sample_distance} km"
        )