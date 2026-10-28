"""Power-delay profiles: 3GPP TS 36.104 Annex B (EPA/EVA/ETU) and TR 38.901 Sec. 7.7.2 (TDL-A, TDL-C)."""

from __future__ import annotations

# delays [ns], powers [dB]
LTE_PROFILES: dict[str, tuple[list[float], list[float]]] = {
    "EPA": ([0, 30, 70, 90, 110, 190, 410], [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]),
    "EVA": ([0, 30, 150, 310, 370, 710, 1090, 1730, 2510], [0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9]),
    "ETU": ([0, 50, 120, 200, 230, 500, 1600, 2300, 5000], [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]),
}

# TR 38.901 Table 7.7.2-1 (TDL-A) and Table 7.7.2-3 (TDL-C): normalized delays and powers [dB]
TDL_NORMALIZED: dict[str, tuple[list[float], list[float]]] = {
    "TDL-A": (
        [0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942,
         2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586],
        [-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7,
         -16.2, -18.3, -18.9, -16.6, -19.9, -29.7],
    ),
    "TDL-C": (
        [0.0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083,
         2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523],
        [-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9,
         -15.8, -17.1, -16.0, -15.7, -21.6, -22.8],
    ),
}


def build_profiles(tdl_delay_spread_ns: float = 300.0) -> dict[str, tuple[list[float], list[float]]]:
    """All profiles with delays in ns; TDL delays are scaled as tau_n = tau_norm * DS (TR 38.901 Eq. 7.7-1)."""
    out = dict(LTE_PROFILES)
    for name, (d, p) in TDL_NORMALIZED.items():
        out[name] = ([x * tdl_delay_spread_ns for x in d], p)
    return out
