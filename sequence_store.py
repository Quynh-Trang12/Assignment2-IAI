import pandas as pd


class SequenceStore:
    """
    Stores ordered traffic flow sequences for each SCATS site.

    Expected dataframe columns:
    - SCATS
    - timestamp
    - flow
    """

    def __init__(self, df: pd.DataFrame):
        self.site_series = {}

        for site_id, group in df.groupby("SCATS"):
            group = group.sort_values("timestamp")
            flows = group["flow"].astype(float).tolist()

            # Need at least 4 values to form one input sequence
            if len(flows) >= 4:
                self.site_series[str(site_id)] = flows

    def get_sequence(self, site_id, t):
        """
        Return the 4 flow values immediately before time index t.

        Example:
        if t = 4, returns flows[0:4]
        if t = 10, returns flows[6:10]

        Returns:
            list of 4 floats, or None if unavailable
        """
        site_id = str(site_id)
        series = self.site_series.get(site_id)

        if series is None:
            return None

        if t < 4:
            return None

        if t > len(series):
            return None

        seq = series[t - 4:t]

        if len(seq) != 4:
            return None

        return seq