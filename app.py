# app.py
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =============================
# Core TEA utilities
# =============================
def crf(wacc: float, n_years: int) -> float:
    if n_years <= 0:
        raise ValueError("project_life must be > 0")
    if wacc == 0:
        return 1 / n_years
    return (wacc * (1 + wacc) ** n_years) / (((1 + wacc) ** n_years) - 1)


def annual_units_from_demand(demand: float, unit: str, capacity_factor: float) -> float:
    """
    unit:
      - "Nm3/h" : demand is average flow
      - "t/yr"  : demand is annual tonnes already
    """
    if unit == "Nm3/h":
        return float(demand) * 8760.0 * float(capacity_factor)
    if unit == "t/yr":
        return float(demand)
    raise ValueError(f"Unsupported unit: {unit}")


def avg_power_kw(flow_nm3h: float, kwh_per_nm3: float) -> float:
    # kWh/Nm3 * Nm3/h = kW
    return float(flow_nm3h) * float(kwh_per_nm3)


def scale_capex(base_capex: float, demand: float, base_demand: float, exponent: float) -> float:
    if demand <= 0:
        return 0.0
    if base_demand <= 0:
        return base_capex
    return base_capex * (demand / base_demand) ** exponent


def lcog(
    annual_units: float,
    capex_installed: float,
    wacc: float,
    project_life: int,
    contingency_frac: float,
    fixed_om_frac: float,
    labor_per_year: float,
    electricity_price: float,
    kwh_per_unit: float,
    var_opex_per_unit: float,
) -> Tuple[float, Dict[str, float]]:
    """
    LCOG = (Annualized CAPEX + Fixed O&M + Labor + Electricity + Variable OPEX) / Annual units
    """
    if annual_units <= 0:
        return float("nan"), {}

    capex_total = capex_installed * (1.0 + contingency_frac)
    annualized_capex = capex_total * crf(wacc, project_life)
    fixed_om = capex_total * fixed_om_frac
    electricity = annual_units * kwh_per_unit * electricity_price
    variable = annual_units * var_opex_per_unit

    total = annualized_capex + fixed_om + labor_per_year + electricity + variable
    br = {
        "Annualized CAPEX": annualized_capex,
        "Fixed O&M": fixed_om,
        "Labor": labor_per_year,
        "Electricity": electricity,
        "Variable OPEX": variable,
        "Total annual": total,
    }
    return total / annual_units, br


# =============================
# Investor-grade UI helpers
# =============================
def inject_css():
    st.markdown(
        """
        <style>
          .wrap {max-width: 1200px; margin: 0 auto;}
          .hero {
            padding: 18px 18px;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(37,99,235,0.08) 0%, rgba(255,255,255,1) 55%);
          }
          .hero h1 {margin: 0; font-size: 24px; color: #0F172A;}
          .hero p {margin: 8px 0 0 0; color: rgba(15,23,42,0.70); font-size: 13px;}
          .pill {
            display:inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(37,99,235,0.10);
            color: #2563EB;
            font-size: 12px;
            font-weight: 700;
            margin-right: 8px;
            border: 1px solid rgba(37,99,235,0.18);
          }
          .kpi {
            padding: 14px 14px;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 14px;
            background: white;
          }
          .kpi .t {font-size: 12px; color: rgba(15,23,42,0.65); margin-bottom: 6px;}
          .kpi .v {font-size: 22px; font-weight: 800; color: #0F172A; line-height: 1.1;}
          .kpi .s {font-size: 12px; color: rgba(15,23,42,0.60); margin-top: 6px;}
          .card {
            padding: 14px 14px;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 14px;
            background: white;
          }
          .section {margin-top: 8px;}
          .muted {color: rgba(15,23,42,0.65);}
          .small {font-size: 12px;}
          .list li {margin-bottom: 6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="t">{title}</div>
          <div class="v">{value}</div>
          <div class="s">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================
# Clean SVG process flow (no matplotlib)
# =============================
def flow_svg(title: str, blocks: List[str], note: str = "", dashed: Optional[List[Tuple[int, int, str]]] = None) -> str:
    dashed = dashed or []
    w, h = 980, 190
    pad = 24
    n = max(1, len(blocks))
    gap = 16
    box_w = int((w - 2 * pad - (n - 1) * gap) / n)
    box_h = 56
    y = 78

    def box(x, txt):
        safe = (
            txt.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return f"""
        <g>
          <rect x="{x}" y="{y}" rx="14" ry="14" width="{box_w}" height="{box_h}"
                fill="white" stroke="rgba(15,23,42,0.18)" stroke-width="1.5"/>
          <text x="{x + box_w/2}" y="{y + 23}" text-anchor="middle" font-size="13" fill="#0F172A"
                font-family="Inter, Arial" font-weight="600">{safe}</text>
        </g>
        """

    def arrow(x1, y1, x2, y2, is_dashed=False):
        style = 'stroke-dasharray="6 6"' if is_dashed else ""
        return f"""<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                 stroke="#0F172A" stroke-width="1.6" {style} marker-end="url(#arrow)"/>"""

    xs = []
    x = pad
    for _ in blocks:
        xs.append(x)
        x += box_w + gap

    arrows = ""
    for i in range(n - 1):
        x1 = xs[i] + box_w
        x2 = xs[i + 1]
        arrows += arrow(x1 + 6, y + box_h / 2, x2 - 6, y + box_h / 2, False)

    dashed_elems = ""
    for (i0, i1, label) in dashed:
        i0 = max(0, min(n - 1, int(i0)))
        i1 = max(0, min(n - 1, int(i1)))
        x1 = xs[i0] + box_w / 2
        x2 = xs[i1] + box_w / 2
        dashed_elems += arrow(x1, y + box_h + 18, x2, y + box_h / 2, True)
        safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        dashed_elems += f"""<text x="{(x1+x2)/2}" y="{y + box_h + 42}" text-anchor="middle"
                          font-size="12" fill="rgba(15,23,42,0.70)" font-family="Inter, Arial">{safe_label}</text>"""

    boxes = "".join([box(xs[i], blocks[i]) for i in range(n)])

    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_note = note.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
    <svg width="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <path d="M0,0 L10,3 L0,6 Z" fill="#0F172A"/>
        </marker>
      </defs>

      <text x="{pad}" y="28" font-size="15" fill="#0F172A" font-family="Inter, Arial" font-weight="800">{safe_title}</text>
      <text x="{pad}" y="50" font-size="12" fill="rgba(15,23,42,0.65)" font-family="Inter, Arial">{safe_note}</text>

      {arrows}
      {boxes}
      {dashed_elems}
    </svg>
    """


def render_flow(title: str, blocks: List[str], note: str = "", dashed: Optional[List[Tuple[int, int, str]]] = None):
    st.markdown(flow_svg(title, blocks, note=note, dashed=dashed), unsafe_allow_html=True)


# =============================
# BOM (range-based quick estimator)
# =============================
def lerp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def interpolate_bom(flow, anchors, components) -> pd.DataFrame:
    anchors = sorted(anchors, key=lambda a: a[0])
    if flow <= anchors[0][0]:
        lo = hi = anchors[0]
    elif flow >= anchors[-1][0]:
        lo = hi = anchors[-1]
    else:
        lo = hi = anchors[-1]
        for a0, a1 in zip(anchors[:-1], anchors[1:]):
            if a0[0] <= flow <= a1[0]:
                lo, hi = a0, a1
                break

    rows = []
    for comp, by_label in components.items():
        low0, high0 = by_label[lo[1]]
        low1, high1 = by_label[hi[1]]
        low = lerp(flow, lo[0], hi[0], low0, low1)
        high = lerp(flow, lo[0], hi[0], high0, high1)
        rows.append([comp, low, high, 0.5 * (low + high)])

    df = pd.DataFrame(rows, columns=["Component", "Low ($)", "High ($)", "Mid ($)"])
    total = df[["Low ($)", "High ($)", "Mid ($)"]].sum()
    df = pd.concat(
        [df, pd.DataFrame([["TOTAL", total["Low ($)"], total["High ($)"], total["Mid ($)"]]], columns=df.columns)],
        ignore_index=True,
    )
    return df


def bom_catalog():
    # These ranges are typical "order-of-magnitude" budgetary buckets. Use as a sanity check / investor story,
    # not as a vendor quote.
    n2_psa_components = {
        "Air compressor package (oil-free preferred)": {"Small": (20000, 60000), "Medium": (60000, 180000), "Large": (200000, 600000)},
        "Aftercooler + moisture separator + drains": {"Small": (2000, 8000), "Medium": (5000, 20000), "Large": (15000, 60000)},
        "Filtration train (particulate + coalescing)": {"Small": (1000, 5000), "Medium": (3000, 12000), "Large": (8000, 30000)},
        "Air dryer (desiccant for reliability)": {"Small": (5000, 25000), "Medium": (15000, 70000), "Large": (40000, 180000)},
        "PSA vessels + adsorbent": {"Small": (15000, 60000), "Medium": (60000, 220000), "Large": (200000, 800000)},
        "Valve manifold + actuators": {"Small": (5000, 25000), "Medium": (20000, 80000), "Large": (60000, 250000)},
        "N2 receiver tank": {"Small": (2000, 10000), "Medium": (8000, 35000), "Large": (25000, 100000)},
        "Analyzer package (O2 ppm / dewpoint)": {"Small": (3000, 15000), "Medium": (5000, 25000), "Large": (10000, 40000)},
        "PLC/HMI + telemetry": {"Small": (5000, 25000), "Medium": (10000, 40000), "Large": (20000, 80000)},
        "Skid integration + wiring": {"Small": (10000, 40000), "Medium": (30000, 120000), "Large": (80000, 300000)},
        "Safety relief/vent routing": {"Small": (1000, 5000), "Medium": (3000, 12000), "Large": (8000, 30000)},
    }
    n2_psa_anchors = [(50.0, "Small"), (250.0, "Medium"), (1000.0, "Large")]

    n2_mem_components = {
        "Air compressor package": {"Small": (15000, 50000), "Medium": (50000, 150000), "Large": (180000, 550000)},
        "Aftercooler + moisture separator": {"Small": (2000, 8000), "Medium": (5000, 20000), "Large": (15000, 60000)},
        "Filtration + dryer": {"Small": (6000, 25000), "Medium": (15000, 60000), "Large": (40000, 180000)},
        "Membrane module(s)": {"Small": (8000, 40000), "Medium": (30000, 140000), "Large": (120000, 450000)},
        "N2 receiver + regulators": {"Small": (3000, 15000), "Medium": (10000, 40000), "Large": (30000, 120000)},
        "Analyzer + PLC/HMI": {"Small": (5000, 30000), "Medium": (12000, 50000), "Large": (25000, 90000)},
        "Skid integration": {"Small": (8000, 35000), "Medium": (25000, 100000), "Large": (70000, 250000)},
    }
    n2_mem_anchors = [(50.0, "Small"), (250.0, "Medium"), (1000.0, "Large")]

    o2_vpsa_components = {
        "Air blower / LP compressor": {"Small": (15000, 60000), "Medium": (40000, 140000), "Large": (120000, 400000)},
        "Aftercooler + filtration": {"Small": (4000, 20000), "Medium": (10000, 45000), "Large": (30000, 120000)},
        "Adsorber vessels + zeolite": {"Small": (25000, 120000), "Medium": (80000, 350000), "Large": (250000, 900000)},
        "Vacuum pump(s) + receiver": {"Small": (20000, 120000), "Medium": (70000, 300000), "Large": (200000, 900000)},
        "Valve manifold + actuators": {"Small": (8000, 40000), "Medium": (25000, 110000), "Large": (80000, 300000)},
        "O2 receiver + regulation": {"Small": (5000, 25000), "Medium": (12000, 60000), "Large": (35000, 160000)},
        "Analyzer + instrumentation": {"Small": (4000, 18000), "Medium": (8000, 30000), "Large": (15000, 45000)},
        "PLC/HMI + telemetry": {"Small": (8000, 35000), "Medium": (15000, 55000), "Large": (25000, 90000)},
        "O2-compatible piping/cleaning": {"Small": (15000, 70000), "Medium": (40000, 160000), "Large": (120000, 450000)},
        "Safety relief/vent routing": {"Small": (2000, 10000), "Medium": (5000, 20000), "Large": (12000, 45000)},
    }
    o2_vpsa_anchors = [(50.0, "Small"), (150.0, "Medium"), (500.0, "Large")]

    return {
        "N2 PSA": (n2_psa_anchors, n2_psa_components),
        "N2 Membrane": (n2_mem_anchors, n2_mem_components),
        "O2 VPSA": (o2_vpsa_anchors, o2_vpsa_components),
    }


def bom_co2_buffer(demand_tpy: float, base_tpy: float = 2000.0, exponent: float = 0.70) -> pd.DataFrame:
    # Rough buffer/storage conditioning BOM bucket
    comps = {
        "Bulk liquid CO2 storage tank(s) + foundations": (60000, 220000),
        "Vaporizer / heater + controls": (12000, 55000),
        "Transfer pump(s) / pressure builder": (8000, 45000),
        "Piping/valves/regulators + install": (20000, 90000),
        "Instrumentation + QA": (8000, 45000),
        "PLC/HMI + telemetry": (8000, 45000),
        "Safety relief / venting / ODH signage": (5000, 25000),
    }
    if demand_tpy <= 0:
        ratio = 0.0
    else:
        ratio = (demand_tpy / base_tpy) ** exponent

    rows = []
    for comp, (lo, hi) in comps.items():
        low = lo * ratio
        high = hi * ratio
        rows.append([comp, low, high, 0.5 * (low + high)])

    df = pd.DataFrame(rows, columns=["Component", "Low ($)", "High ($)", "Mid ($)"])
    total = df[["Low ($)", "High ($)", "Mid ($)"]].sum()
    df = pd.concat(
        [df, pd.DataFrame([["TOTAL", total["Low ($)"], total["High ($)"], total["Mid ($)"]]], columns=df.columns)],
        ignore_index=True,
    )
    return df


# =============================
# Technology library (defaults tuned)
# =============================
def tech_library() -> pd.DataFrame:
    """
    Defaults are chosen to be plausible and easy to defend:
      - N2 PSA kWh/Nm3 in the ~0.3–0.6 band (we set mid default).
      - O2 low-pressure VPSA in the ~0.35–0.40 band; high pressure adds booster penalty.
      - Delivered prices are placeholders: set per-customer contract in app.
    """
    cols = [
        "gas",
        "purity_tier",
        "unit",
        "option",
        "category",
        "trl",
        "capex_installed",
        "kwh_per_unit",
        "var_opex_per_unit",
        "fixed_om_frac",
        "labor_per_year",
        "base_demand_value",
        "capex_scale_exp",
        "why",
        "best_fit",
        "watchouts",
        "bom_family",
    ]

    rows = [
        # ---------------- N2 ----------------
        ["Nitrogen (N2)", "Industrial (99.9%)", "Nm3/h",
         "Incumbent: LIN delivered + tank", "Incumbent", 9,
         150000, 0.00, 0.22, 0.02, 5000,
         250.0, 0.70,
         "Fast deployment and simple operations.",
         "Small or temporary demand; highly variable usage.",
         "Cost and uptime tied to deliveries and contract terms.",
         ""],

        ["Nitrogen (N2)", "Industrial (99.9%)", "Nm3/h",
         "Platform: PSA N2 (99.9%)", "Platform", 9,
         450000, 0.45, 0.00, 0.05, 12000,
         250.0, 0.70,
         "Deletes trucking. Predictable OPEX. Good economics at steady demand.",
         "Steady industrial demand (heat treat, metals, glass, packaging).",
         "Needs clean dry air and maintenance; purity tied to media health.",
         "N2 PSA"],

        ["Nitrogen (N2)", "Industrial (99.9%)", "Nm3/h",
         "Platform: Membrane N2 (95–99%)", "Platform", 9,
         300000, 0.30, 0.00, 0.04, 9000,
         250.0, 0.70,
         "Lower CAPEX and simpler than PSA; quick install.",
         "Smaller sites that can tolerate lower purity.",
         "Purity/recovery tradeoff; not ideal for tight specs.",
         "N2 Membrane"],

        ["Nitrogen (N2)", "High purity (5N)", "Nm3/h",
         "Incumbent: LIN delivered + tank (5N contract)", "Incumbent", 9,
         150000, 0.00, 0.28, 0.02, 5000,
         250.0, 0.70,
         "Easiest path to high purity with minimal onsite complexity.",
         "Any high purity site without operator appetite.",
         "Expensive per unit; logistics and volatility remain.",
         ""],

        ["Nitrogen (N2)", "High purity (5N)", "Nm3/h",
         "Platform: PSA + purifier (5N)", "Platform", 9,
         800000, 0.55, 0.00, 0.05, 16000,
         250.0, 0.70,
         "High purity onsite while reducing logistics dependency.",
         "High-purity inerting and specialty processing at steady loads.",
         "Higher CAPEX and power; purifier adds complexity.",
         "N2 PSA"],

        # ---------------- O2 ----------------
        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Incumbent: LOX delivered + tank", "Incumbent", 9,
         200000, 0.00, 0.30, 0.02, 5000,
         150.0, 0.70,
         "Simple: buy oxygen, consume oxygen.",
         "Low/moderate or very peaky demand.",
         "Logistics heavy; supply disruptions happen.",
         ""],

        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Platform: VPSA O2 (low pressure)", "Platform", 9,
         700000, 0.36, 0.00, 0.05, 18000,
         150.0, 0.70,
         "Cheapest onsite O2 mode: keep pressure low, avoid booster penalty.",
         "Glass/metals/wastewater enrichment at low pressure.",
         "If you need higher pressure, booster adds cost + kWh.",
         "O2 VPSA"],

        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Platform: VPSA O2 + booster (6 barg)", "Platform", 9,
         900000, 0.55, 0.00, 0.05, 20000,
         150.0, 0.70,
         "Onsite O2 compatible with higher-pressure distribution.",
         "Sites with pressurized O2 header.",
         "Booster is the expensive lever; try to avoid if process allows.",
         "O2 VPSA"],

        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Platform: Hybrid (VPSA base-load + LOX peaks)", "Platform", 9,
         650000, 0.36, 0.00, 0.05, 18000,
         150.0, 0.70,
         "Lower CAPEX (don’t oversize). LOX covers peaks and maintenance windows.",
         "Customers who want uptime without oversizing.",
         "Still needs LOX contract/tank; economics depend on peak fraction.",
         "O2 VPSA"],

        ["Oxygen (O2)", "High purity (99.5%+)", "Nm3/h",
         "Incumbent: LOX delivered + tank (high purity)", "Incumbent", 9,
         200000, 0.00, 0.33, 0.02, 5000,
         150.0, 0.70,
         "High purity supply without owning a plant.",
         "Sites that require high purity but don’t want cryo complexity.",
         "Logistics risk remains; cost higher.",
         ""],

        ["Oxygen (O2)", "High purity (99.5%+)", "Nm3/h",
         "Platform: micro-cryo ASU (99.5%+)", "Platform", 9,
         3500000, 0.90, 0.00, 0.05, 40000,
         150.0, 0.70,
         "Onsite high purity where purity and SLA justify it.",
         "High-purity must-have sites with steady demand.",
         "High CAPEX and complexity; typically bigger sites.",
         ""],

        # ---------------- CO2 ----------------
        ["Carbon Dioxide (CO2)", "Industrial", "t/yr",
         "Incumbent: LCO2 delivered + tank (industrial)", "Incumbent", 9,
         120000, 0.00, 350.0, 0.02, 4000,
         2000.0, 0.70,
         "Simplest when supply is stable.",
         "Non-critical industrial uses.",
         "Shortage events can shut down production.",
         ""],

        ["Carbon Dioxide (CO2)", "Food/Bev", "t/yr",
         "Incumbent: LCO2 delivered + tank (food/bev)", "Incumbent", 9,
         120000, 0.00, 450.0, 0.02, 4000,
         2000.0, 0.70,
         "QA handled by supplier.",
         "Food/bev where QA matters.",
         "Availability and pricing volatility.",
         ""],

        ["Carbon Dioxide (CO2)", "Industrial", "t/yr",
         "Platform: Conditioning + storage (no capture)", "Platform", 9,
         400000, 15.0, 0.00, 0.05, 12000,
         2000.0, 0.70,
         "You’re selling uptime: buffer + conditioning + telemetry + SLA.",
         "Customers burned by outages.",
         "Does not generate CO2; still needs a source contract.",
         ""],

        ["Carbon Dioxide (CO2)", "Food/Bev", "t/yr",
         "Platform: Conditioning + storage (no capture)", "Platform", 9,
         450000, 18.0, 0.00, 0.05, 14000,
         2000.0, 0.70,
         "Same story as industrial, but QA-focused conditioning.",
         "Food/bev where outages are catastrophic.",
         "Does not generate CO2; still needs a source contract.",
         ""],
    ]

    return pd.DataFrame(rows, columns=cols)


# =============================
# Process flow specs
# =============================
def flow_for(gas: str, option: str) -> Tuple[List[str], str, List[Tuple[int, int, str]]]:
    g = gas.lower()
    o = option.lower()

    if "nitrogen" in g:
        if "delivered" in o:
            return (
                ["LIN delivery", "Bulk LIN tank", "Vaporizer", "Regulation", "N₂ header", "Point-of-use"],
                "Delivered model: simplest ops; cost + uptime tied to logistics.",
                [(1, 4, "Buffer inventory")],
            )
        if "membrane" in o:
            return (
                ["Ambient air", "Compressor", "Dryer/filters", "Membrane", "Receiver", "N₂ header"],
                "Membrane: lower CAPEX; purity/recovery tradeoff.",
                [(3, 3, "O₂-rich permeate → vent")],
            )
        if "purifier" in o or "5n" in o:
            return (
                ["Ambient air", "Compressor", "Dryer/filters", "PSA beds", "Purifier", "Receiver", "N₂ header"],
                "PSA + purifier: higher purity; higher CAPEX + power.",
                [(3, 3, "Offgas → vent")],
            )
        return (
            ["Ambient air", "Compressor", "Dryer/filters", "PSA beds", "Receiver", "N₂ header"],
            "PSA: strong economics at steady demand.",
            [(3, 3, "Offgas → vent")],
        )

    if "oxygen" in g:
        if "delivered" in o:
            return (
                ["LOX delivery", "Bulk LOX tank", "Vaporizer", "Regulation", "O₂ receiver", "O₂ header"],
                "Delivered model: simple, logistics-heavy.",
                [(1, 4, "Buffer inventory")],
            )
        if "hybrid" in o:
            return (
                ["Ambient air", "VPSA base-load", "O₂ receiver", "O₂ header", "LOX peak supply", "SLA monitoring"],
                "Hybrid: smaller VPSA for base-load + LOX for peaks/maintenance windows.",
                [(4, 3, "Peak/backup feed")],
            )
        if "cryo" in o or "asu" in o:
            return (
                ["Ambient air", "Compressor", "Pre-purification", "Cold box", "O₂ storage", "O₂ header"],
                "Micro-cryo ASU: high purity; high CAPEX + complexity.",
                [(3, 3, "N₂/Ar coproducts (optional)")],
            )
        return (
            ["Ambient air", "Blower/LP comp", "Filters", "VPSA beds + vacuum", "O₂ receiver", "O₂ header"],
            "VPSA: best economics at low product pressure; avoid booster penalty if possible.",
            [(3, 3, "N₂-rich tailgas → vent")],
        )

    if "carbon dioxide" in g or "co2" in g:
        if "delivered" in o:
            return (
                ["LCO₂ delivery", "Bulk tank", "Pump/pressure build", "Vaporizer", "QA", "CO₂ header"],
                "Delivered CO₂ is simple when supply is stable; vulnerable to shortage events.",
                [(1, 5, "Buffer inventory")],
            )
        return (
            ["Supply contract/source", "Bulk storage", "Conditioning", "Telemetry + QA", "CO₂ header", "SLA monitoring"],
            "Platform CO₂ here is uptime/buffering/monitoring (not CO₂ generation).",
            [(1, 4, "Extended buffer")],
        )

    return (["Input", "Process", "Output"], "Flow not defined yet.", [])


# =============================
# App
# =============================
st.set_page_config(page_title="Gas Utility Platform TEA", page_icon="🧪", layout="wide")
inject_css()

tech = tech_library()

st.markdown('<div class="wrap">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <h1>Modular On-Site Gas Utility Platform</h1>
      <p>Pick one gas, size the requirement, compare incumbent vs platform, and see the economics + the story investors ask for.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar controls
with st.sidebar:
    st.markdown("### Controls")
    investor_mode = st.toggle("Investor Mode", value=True)
    st.caption("Investor Mode hides most spreadsheet detail and leads with the narrative + visuals.")

    st.divider()
    gas = st.selectbox("Gas", options=tech["gas"].unique().tolist(), index=0)

    purity_options = tech[tech["gas"] == gas]["purity_tier"].unique().tolist()
    purity = st.selectbox("Purity tier", options=purity_options, index=0)

    subset = tech[(tech["gas"] == gas) & (tech["purity_tier"] == purity)].copy().reset_index(drop=True)
    unit = subset["unit"].iloc[0]

    st.divider()
    st.markdown("### Assumptions")
    electricity_price = st.number_input("Electricity ($/kWh)", value=0.10, step=0.01, format="%.3f")
    wacc = st.number_input("WACC", value=0.12, step=0.01, format="%.3f")
    project_life = st.number_input("Project life (years)", value=10, step=1)
    capacity_factor = st.number_input("Capacity factor (for flow)", value=0.90, step=0.01, format="%.3f")
    contingency_frac = st.number_input("Contingency (fraction)", value=0.15, step=0.01, format="%.3f")
    capex_scale_exp = st.number_input("CAPEX scaling exponent", value=0.70, step=0.05, format="%.2f")

    st.divider()
    st.markdown("### Company view")
    integration_factor = st.number_input("Integration factor (BOM→build)", value=1.30, step=0.05, format="%.2f")
    install_factor = st.number_input("Install factor (build→installed)", value=1.25, step=0.05, format="%.2f")
    target_gm = st.number_input("Target gross margin", value=0.35, step=0.05, format="%.2f")

    st.divider()
    st.markdown("### Hybrid controls")
    peak_frac = st.slider("Hybrid peak fraction (delivered share)", 0.0, 0.60, 0.20, 0.05)
    lox_backup_capex = st.number_input("Hybrid LOX backup CAPEX add ($)", value=150000.0, step=25000.0)

# ---- Demand input
st.markdown("<div class='section'>", unsafe_allow_html=True)
tag = f"<span class='pill'>{gas}</span><span class='pill'>{purity}</span>"
st.markdown(tag, unsafe_allow_html=True)

default_demand = float(subset["base_demand_value"].iloc[0])
if unit == "Nm3/h":
    demand = st.number_input("Average flow (Nm³/h)", value=default_demand, step=10.0)
else:
    demand = st.number_input("Annual demand (t/yr)", value=default_demand, step=100.0)

annual_u = annual_units_from_demand(demand, unit, capacity_factor)
unit_label = "$/Nm³" if unit == "Nm3/h" else "$/t"

st.caption(
    f"Annual units: **{annual_u:,.0f}** "
    f"{'Nm³/yr' if unit=='Nm3/h' else 't/yr'}. LCOG is reported in **{unit_label}**."
)

# ---- Option selection
inc_opts = subset[subset["category"] == "Incumbent"]["option"].tolist()
plat_opts = subset[subset["category"] == "Platform"]["option"].tolist()

c1, c2 = st.columns(2)
with c1:
    incumbent_choice = st.selectbox("Incumbent option", options=inc_opts, index=0)
with c2:
    platform_choice = st.selectbox("Platform option", options=plat_opts, index=0)

inc = subset[subset["option"] == incumbent_choice].iloc[0].to_dict()
plat = subset[subset["option"] == platform_choice].iloc[0].to_dict()

# ---- Advanced tuning (optional)
with st.expander("Advanced: tune the scenario inputs (delivered price, kWh, CAPEX)", expanded=False):
    st.write("Use this to match a specific customer contract or a vendor quote.")
    a1, a2, a3 = st.columns(3)
    with a1:
        inc_var_price = st.number_input("Incumbent variable price ($/unit)", value=float(inc["var_opex_per_unit"]), step=0.01 if unit == "Nm3/h" else 10.0)
        inc_capex_base = st.number_input("Incumbent base CAPEX ($)", value=float(inc["capex_installed"]), step=25000.0)
    with a2:
        plat_var_price = st.number_input("Platform variable OPEX ($/unit)", value=float(plat["var_opex_per_unit"]), step=0.01 if unit == "Nm3/h" else 10.0)
        plat_capex_base = st.number_input("Platform base CAPEX ($)", value=float(plat["capex_installed"]), step=25000.0)
    with a3:
        plat_kwh_override = st.number_input("Platform kWh/unit override", value=float(plat["kwh_per_unit"]), step=0.01)
    inc["var_opex_per_unit"] = inc_var_price
    inc["capex_installed"] = inc_capex_base
    plat["var_opex_per_unit"] = plat_var_price
    plat["capex_installed"] = plat_capex_base
    plat["kwh_per_unit"] = plat_kwh_override

# ---- Oxygen “cheap route” lever: pressure profile (only meaningful for O2)
# (These values are defaults that keep the story defensible: low pressure is cheaper; booster adds a penalty.)
O2_PROFILES = {
    "Low pressure (no booster)": 0.36,
    "Pressurized header (booster penalty)": 0.55,
    "Conservative / worst-case": 0.75,
}
if gas.startswith("Oxygen") and "VPSA" in platform_choice:
    prof = st.selectbox("O₂ delivery pressure profile", options=list(O2_PROFILES.keys()), index=0)
    plat["kwh_per_unit"] = float(O2_PROFILES[prof])
    if "Pressurized" in prof:
        st.info("Cost lever: if the process can accept low-pressure O₂, you avoid the booster penalty and the economics improve.")
    if "Conservative" in prof:
        st.warning("This is a conservative energy assumption for investor downside scenarios.")

# ---- CAPEX scaling
inc_capex = scale_capex(float(inc["capex_installed"]), float(demand), float(inc["base_demand_value"]), capex_scale_exp)
plat_capex = scale_capex(float(plat["capex_installed"]), float(demand), float(plat["base_demand_value"]), capex_scale_exp)

# Hybrid logic (O2 only): resize VPSA for base-load; buy delivered for peaks; add LOX backup capex add
if gas.startswith("Oxygen") and "Hybrid" in platform_choice:
    base_flow = (1.0 - peak_frac) * float(demand)
    plat_capex = scale_capex(float(plat["capex_installed"]), base_flow, float(plat["base_demand_value"]), capex_scale_exp) + float(lox_backup_capex)
    # Blend: base supplied by plant, peaks supplied by delivered LOX (proxy using incumbent variable price)
    plat["var_opex_per_unit"] = peak_frac * float(inc["var_opex_per_unit"])

# ---- TEA compute
inc_lcog, inc_br = lcog(
    annual_units=annual_u,
    capex_installed=inc_capex,
    wacc=float(wacc),
    project_life=int(project_life),
    contingency_frac=float(contingency_frac),
    fixed_om_frac=float(inc["fixed_om_frac"]),
    labor_per_year=float(inc["labor_per_year"]),
    electricity_price=float(electricity_price),
    kwh_per_unit=float(inc["kwh_per_unit"]),
    var_opex_per_unit=float(inc["var_opex_per_unit"]),
)
plat_lcog, plat_br = lcog(
    annual_units=annual_u,
    capex_installed=plat_capex,
    wacc=float(wacc),
    project_life=int(project_life),
    contingency_frac=float(contingency_frac),
    fixed_om_frac=float(plat["fixed_om_frac"]),
    labor_per_year=float(plat["labor_per_year"]),
    electricity_price=float(electricity_price),
    kwh_per_unit=float(plat["kwh_per_unit"]),
    var_opex_per_unit=float(plat["var_opex_per_unit"]),
)

inc_annual = inc_br.get("Total annual", float("nan"))
plat_annual = plat_br.get("Total annual", float("nan"))
annual_savings = inc_annual - plat_annual if (not math.isnan(inc_annual) and not math.isnan(plat_annual)) else float("nan")

inc_capex_total = inc_capex * (1 + contingency_frac)
plat_capex_total = plat_capex * (1 + contingency_frac)
delta_capex = plat_capex_total - inc_capex_total
payback = (delta_capex / annual_savings) if (annual_savings and annual_savings > 0 and delta_capex > 0) else float("nan")

# power sanity for flow gases
inc_kw = avg_power_kw(demand, float(inc["kwh_per_unit"])) if unit == "Nm3/h" else 0.0
plat_kw = avg_power_kw(demand, float(plat["kwh_per_unit"])) if unit == "Nm3/h" else 0.0

# ---- Executive KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Annual units", f"{annual_u:,.0f}", "Nm³/yr" if unit == "Nm3/h" else "t/yr")
with k2:
    kpi_card("Incumbent LCOG", "n/a" if math.isnan(inc_lcog) else f"${inc_lcog:,.3f}", unit_label)
with k3:
    kpi_card("Platform LCOG", "n/a" if math.isnan(plat_lcog) else f"${plat_lcog:,.3f}", unit_label)
with k4:
    if not math.isnan(inc_lcog) and not math.isnan(plat_lcog) and inc_lcog > 0:
        kpi_card("Savings", f"{(1 - plat_lcog / inc_lcog) * 100:,.1f}%", "vs incumbent")
    else:
        kpi_card("Savings", "n/a", "")

k5, k6, k7, k8 = st.columns(4)
with k5:
    kpi_card("Inc CAPEX (+cont)", f"${inc_capex_total:,.0f}", "")
with k6:
    kpi_card("Plat CAPEX (+cont)", f"${plat_capex_total:,.0f}", "")
with k7:
    kpi_card("Annual savings", "n/a" if math.isnan(annual_savings) else f"${annual_savings:,.0f}/yr", "")
with k8:
    kpi_card("Payback", "n/a" if math.isnan(payback) else f"{payback:,.1f} yrs", "")

if unit == "Nm3/h":
    st.caption(f"Power sanity: Inc ~{inc_kw:,.1f} kW | Platform ~{plat_kw:,.1f} kW (kWh/Nm³ × Nm³/h).")

# ---- Tabs
tab_overview, tab_econ, tab_process, tab_company, tab_sens, tab_sources = st.tabs(
    ["Overview", "Economics", "Process flow", "Company view", "Sensitivity", "Sources"]
)

with tab_overview:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Why this platform can be better**")
    st.write(f"• {plat['why']}")
    st.write(f"• Best fit: {plat['best_fit']}")
    st.write(f"• Watch-outs: {plat['watchouts']}")
    if gas.startswith("Oxygen") and "VPSA" in platform_choice:
        st.write("• Key cost lever for O₂: avoid pressurizing the whole stream if the process can accept low-pressure oxygen.")
    if gas.startswith("Oxygen") and "Hybrid" in platform_choice:
        st.write("• Hybrid lever: size the plant for base-load and use delivered LOX for peaks/maintenance windows.")
    if gas.startswith("Carbon Dioxide") and "no capture" in platform_choice.lower():
        st.write("• CO₂ platform option here is resilience-focused: buffering + telemetry + SLA, not CO₂ generation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Quick comparison charts")
    summary_df = pd.DataFrame(
        {
            "Scenario": ["Incumbent", "Platform"],
            "LCOG": [inc_lcog, plat_lcog],
            "Annual cost ($/yr)": [inc_annual, plat_annual],
            "CAPEX+cont ($)": [inc_capex_total, plat_capex_total],
        }
    ).set_index("Scenario")

    cA, cB = st.columns(2)
    with cA:
        st.bar_chart(summary_df[["LCOG"]])
    with cB:
        st.bar_chart(summary_df[["Annual cost ($/yr)"]])

with tab_econ:
    st.markdown("#### Cost breakdown")
    breakdown = pd.DataFrame(
        [
            ["Annualized CAPEX", inc_br.get("Annualized CAPEX", 0), plat_br.get("Annualized CAPEX", 0)],
            ["Fixed O&M", inc_br.get("Fixed O&M", 0), plat_br.get("Fixed O&M", 0)],
            ["Labor", inc_br.get("Labor", 0), plat_br.get("Labor", 0)],
            ["Electricity", inc_br.get("Electricity", 0), plat_br.get("Electricity", 0)],
            ["Variable OPEX", inc_br.get("Variable OPEX", 0), plat_br.get("Variable OPEX", 0)],
            ["Total annual", inc_br.get("Total annual", 0), plat_br.get("Total annual", 0)],
        ],
        columns=["Cost item", "Incumbent ($/yr)", "Platform ($/yr)"],
    )
    if not investor_mode:
        st.dataframe(breakdown, use_container_width=True)
    else:
        st.dataframe(breakdown.iloc[:-1], use_container_width=True)

    st.markdown("#### Payback curve")
    if not math.isnan(annual_savings) and annual_savings > 0 and delta_capex > 0:
        years = np.arange(0, int(project_life) + 1)
        cum = -delta_capex + annual_savings * years
        st.line_chart(pd.DataFrame({"Cumulative cashflow ($)": cum}, index=years))
        st.caption("Year 0 is the extra CAPEX; each year adds the annual savings.")
    else:
        st.info("Payback not shown because savings are not positive or platform CAPEX is not higher.")

with tab_process:
    st.markdown("#### Incumbent vs Platform process flow")
    inc_blocks, inc_note, inc_dashed = flow_for(gas, incumbent_choice)
    plat_blocks, plat_note, plat_dashed = flow_for(gas, platform_choice)

    cA, cB = st.columns(2)
    with cA:
        render_flow("Incumbent", inc_blocks, note=inc_note, dashed=inc_dashed)
    with cB:
        render_flow("Platform", plat_blocks, note=plat_note, dashed=plat_dashed)

with tab_company:
    st.markdown("#### System cost (company view)")
    st.caption("This is a budgetary cost story: BOM mid → build → installed → suggested sale price.")

    bom_df = None
    bom_family = str(plat.get("bom_family", "") or "")

    if unit == "Nm3/h" and bom_family in ("N2 PSA", "N2 Membrane", "O2 VPSA"):
        anchors, comps = bom_catalog()[bom_family]
        bom_df = interpolate_bom(float(demand), anchors, comps)

    if gas.startswith("Carbon Dioxide"):
        bom_df = bom_co2_buffer(float(demand), base_tpy=float(plat["base_demand_value"]), exponent=float(capex_scale_exp))

    if bom_df is None:
        st.info("No BOM model attached to this selection.")
    else:
        st.dataframe(bom_df, use_container_width=True)
        bom_mid = float(bom_df[bom_df["Component"] == "TOTAL"]["Mid ($)"].values[0])

        build_cost = bom_mid * float(integration_factor)
        installed_cost = build_cost * float(install_factor)
        installed_with_cont = installed_cost * (1 + float(contingency_frac))
        sale_price = installed_with_cont / max(1e-6, (1 - float(target_gm)))

        company = pd.DataFrame(
            [
                ["BOM mid (equipment)", bom_mid],
                ["Build cost (BOM × integration)", build_cost],
                ["Installed cost (build × install)", installed_cost],
                ["Installed + contingency", installed_with_cont],
                ["Suggested sale price (target GM)", sale_price],
            ],
            columns=["Item", "$"],
        )
        st.dataframe(company, use_container_width=True)

        st.markdown("#### How to make O₂ cheaper (what to tell customers + VCs)")
        if gas.startswith("Oxygen"):
            st.write("• Keep product pressure low when possible: boosters are the cost villain.")
            st.write("• Use hybrid: VPSA sized for base-load + LOX for peaks/maintenance windows.")
            st.write("• Sell industrial purity (90–93%) unless the process truly needs 99.5%+.")

with tab_sens:
    st.markdown("#### Sensitivity (what flips the decision)")
    st.caption("Two fast levers: electricity price and delivered-gas unit price. This is the investor question: when do we win?")

    if unit == "Nm3/h":
        # Sweep electricity price
        ep = np.linspace(0.05, 0.25, 25)
        rows = []
        for e in ep:
            _, i_br = lcog(annual_u, inc_capex, wacc, int(project_life), contingency_frac, float(inc["fixed_om_frac"]),
                           float(inc["labor_per_year"]), e, float(inc["kwh_per_unit"]), float(inc["var_opex_per_unit"]))
            _, p_br = lcog(annual_u, plat_capex, wacc, int(project_life), contingency_frac, float(plat["fixed_om_frac"]),
                           float(plat["labor_per_year"]), e, float(plat["kwh_per_unit"]), float(plat["var_opex_per_unit"]))
            rows.append([e, i_br["Total annual"], p_br["Total annual"]])

        sdf = pd.DataFrame(rows, columns=["Electricity ($/kWh)", "Incumbent $/yr", "Platform $/yr"]).set_index("Electricity ($/kWh)")
        st.line_chart(sdf)

        # Delivered price multiplier sweep (keep platform fixed)
        st.markdown("#### Delivered price multiplier (contract swing)")
        mult = np.linspace(0.6, 1.6, 21)
        rows2 = []
        base_price = float(inc["var_opex_per_unit"])
        for m in mult:
            _, i_br = lcog(annual_u, inc_capex, wacc, int(project_life), contingency_frac, float(inc["fixed_om_frac"]),
                           float(inc["labor_per_year"]), electricity_price, float(inc["kwh_per_unit"]), base_price * m)
            rows2.append([m, i_br["Total annual"], plat_annual])

        sdf2 = pd.DataFrame(rows2, columns=["Delivered price x", "Incumbent $/yr", "Platform $/yr"]).set_index("Delivered price x")
        st.line_chart(sdf2)

    else:
        st.info("Sensitivity is most useful for flow-based gases. For CO₂, delivered contract terms and outage cost dominate.")

with tab_sources:
    st.markdown("#### Sources and grounding (for investor diligence)")
    st.write("We keep the defaults conservative and editable. For investor decks, cite the broad band and show sensitivity rather than claiming a single ‘true’ number.")
    st.markdown("**Public reference links (pasteable):**")
    st.code(
        "\n".join([
            "DOE TECHTEST overview: https://www.energy.gov/eere/ito/techno-economic-heuristic-tool-early-stage-technologies-techtest-tool",
            "Messer PSA nitrogen (mentions low specific power 0.3–0.6 kWh/Nm³): https://applications.messergroup.com/psa",
            "Messer VPSA oxygen (example low specific power ~0.36 kWh/Nm³ at low pressure): https://applications.messergroup.com/oxygen_generator/vpsa",
            "PATH/CHAI oxygen brief (VSA vs PSA energy context): https://media.path.org/documents/O2_generation_and_storage_PSA_VSA_v1.pdf",
        ]),
        language="text",
    )
    st.caption("If you want, we can add an internal ‘source note’ column per technology row and show citations inline in the UI.")

# ---- Export
st.markdown("### Export")
export_df = pd.DataFrame([
    {
        "Gas": gas,
        "Purity tier": purity,
        "Scenario": "Incumbent",
        "Option": incumbent_choice,
        "Demand": demand,
        "Demand unit": unit,
        "Annual units": annual_u,
        "CAPEX installed ($)": inc_capex,
        "CAPEX+cont ($)": inc_capex_total,
        "kWh/unit": float(inc["kwh_per_unit"]),
        "Var OPEX ($/unit)": float(inc["var_opex_per_unit"]),
        "LCOG": inc_lcog,
        "Annual cost ($/yr)": inc_annual,
        "Avg power (kW)": inc_kw,
        "TRL": int(inc["trl"]),
    },
    {
        "Gas": gas,
        "Purity tier": purity,
        "Scenario": "Platform",
        "Option": platform_choice,
        "Demand": demand,
        "Demand unit": unit,
        "Annual units": annual_u,
        "CAPEX installed ($)": plat_capex,
        "CAPEX+cont ($)": plat_capex_total,
        "kWh/unit": float(plat["kwh_per_unit"]),
        "Var OPEX ($/unit)": float(plat["var_opex_per_unit"]),
        "LCOG": plat_lcog,
        "Annual cost ($/yr)": plat_annual,
        "Avg power (kW)": plat_kw,
        "TRL": int(plat["trl"]),
    },
])

st.download_button(
    "Download results.csv",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="gas_utility_tea_results.csv",
    mime="text/csv",
)

st.markdown("</div>", unsafe_allow_html=True)
