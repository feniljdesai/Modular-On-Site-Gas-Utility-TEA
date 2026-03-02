import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

if gas.startswith("Oxygen") and demand < 400:
    st.info("At <400 Nm³/h, consider: (1) delivered LOX, or (2) Hybrid VPSA base-load + LOX peaks, to avoid small-system CAPEX penalty.")
# -----------------------------
# Finance + TEA utilities
# -----------------------------
def crf(wacc: float, n_years: int) -> float:
    if n_years <= 0:
        raise ValueError("Project life must be > 0")
    if wacc == 0:
        return 1 / n_years
    return (wacc * (1 + wacc) ** n_years) / (((1 + wacc) ** n_years) - 1)


def annual_units(demand: float, unit: str, capacity_factor: float) -> float:
    if unit == "Nm3/h":
        return float(demand) * 8760.0 * capacity_factor
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
    annual_units_value: float,
    capex_installed: float,
    wacc: float,
    project_life: int,
    contingency_pct: float,
    fixed_om_pct: float,
    labor_per_year: float,
    electricity_price: float,
    kwh_per_unit: float,
    variable_opex_per_unit: float,
) -> Tuple[float, Dict[str, float]]:
    if annual_units_value <= 0:
        return float("nan"), {}

    capex_total = capex_installed * (1.0 + contingency_pct)
    annualized_capex = capex_total * crf(wacc, project_life)
    fixed_om = capex_total * fixed_om_pct
    electricity = annual_units_value * kwh_per_unit * electricity_price
    variable_opex = annual_units_value * variable_opex_per_unit
    total_annual = annualized_capex + fixed_om + labor_per_year + electricity + variable_opex

    breakdown = {
        "Annualized CAPEX": annualized_capex,
        "Fixed O&M": fixed_om,
        "Labor": labor_per_year,
        "Electricity": electricity,
        "Variable OPEX": variable_opex,
        "Total annual": total_annual,
    }
    return total_annual / annual_units_value, breakdown


# -----------------------------
# Investor-grade UI helpers (CSS + cards)
# -----------------------------
def inject_css():
    st.markdown(
        """
        <style>
          .kpi-card {
            padding: 14px 16px;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 14px;
            background: white;
          }
          .kpi-title {font-size: 12px; color: rgba(15,23,42,0.65); margin-bottom: 6px;}
          .kpi-value {font-size: 22px; font-weight: 700; color: #0F172A; line-height: 1.1;}
          .kpi-sub {font-size: 12px; color: rgba(15,23,42,0.60); margin-top: 6px;}
          .pill {
            display:inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(37,99,235,0.10);
            color: #2563EB;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
          }
          .section-title {font-size: 16px; font-weight: 750; margin-top: 6px;}
          .muted {color: rgba(15,23,42,0.65);}
          .box {
            padding: 14px 16px;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 14px;
            background: #FFFFFF;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Clean SVG flow diagram (looks professional, no deps)
# -----------------------------
def flow_svg(title: str, blocks: List[str], note: str = "", dashed: List[Tuple[int, int, str]] = None) -> str:
    dashed = dashed or []
    w = 980
    h = 170
    pad = 26
    n = len(blocks)
    box_w = int((w - 2 * pad - (n - 1) * 18) / n)
    box_h = 54
    y = 64

    def box(x, txt):
        return f"""
        <g>
          <rect x="{x}" y="{y}" rx="14" ry="14" width="{box_w}" height="{box_h}"
                fill="white" stroke="rgba(15,23,42,0.18)" stroke-width="1.5"/>
          <text x="{x + box_w/2}" y="{y + 22}" text-anchor="middle" font-size="13" fill="#0F172A" font-family="Inter, Arial">
            {txt}
          </text>
        </g>
        """

    def arrow(x1, y1, x2, y2, dashed=False):
        style = 'stroke-dasharray="6 6"' if dashed else ""
        return f"""
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#0F172A" stroke-width="1.6" {style} marker-end="url(#arrow)"/>
        """

    x_positions = []
    x = pad
    for _ in blocks:
        x_positions.append(x)
        x += box_w + 18

    # main arrows
    arrows = ""
    for i in range(n - 1):
        x1 = x_positions[i] + box_w
        x2 = x_positions[i + 1]
        arrows += arrow(x1 + 6, y + box_h / 2, x2 - 6, y + box_h / 2, dashed=False)

    # dashed annotations
    dashed_elems = ""
    for (i0, i1, label) in dashed:
        x1 = x_positions[i0] + box_w / 2
        x2 = x_positions[i1] + box_w / 2
        dashed_elems += arrow(x1, y + box_h + 18, x2, y + box_h / 2, dashed=True)
        dashed_elems += f"""<text x="{(x1+x2)/2}" y="{y + box_h + 40}" text-anchor="middle" font-size="12" fill="rgba(15,23,42,0.70)" font-family="Inter, Arial">{label}</text>"""

    boxes = "".join([box(x_positions[i], blocks[i]) for i in range(n)])

    svg = f"""
    <svg width="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <path d="M0,0 L10,3 L0,6 Z" fill="#0F172A"/>
        </marker>
      </defs>

      <text x="{pad}" y="28" font-size="15" fill="#0F172A" font-family="Inter, Arial" font-weight="700">{title}</text>
      <text x="{pad}" y="48" font-size="12" fill="rgba(15,23,42,0.65)" font-family="Inter, Arial">{note}</text>

      {arrows}
      {boxes}
      {dashed_elems}
    </svg>
    """
    return svg


def render_flow(title: str, blocks: List[str], note: str = "", dashed=None):
    st.markdown(flow_svg(title, blocks, note=note, dashed=dashed), unsafe_allow_html=True)


# -----------------------------
# Tech library (updated to align with common published energy ranges)
# -----------------------------
def tech_library() -> pd.DataFrame:
    """
    Notes on numbers:
    - O2 VPSA/VSA low-pressure kWh/Nm3 is commonly reported ~0.29–0.39 depending on design/conditions.
    - N2 PSA kWh/Nm3 reported ~0.3–0.6 depending on purity + pressure + compressor efficiency.
    (We still keep price/contract as user-adjustable because delivered-gas pricing swings wildly.)
    """
    cols = [
        "gas", "purity_tier", "unit",
        "option", "category", "trl",
        "capex_installed", "kwh_per_unit", "var_opex_per_unit",
        "fixed_om_pct", "labor_per_year",
        "base_demand_value", "capex_scale_exp",
        "why", "watchouts", "best_fit",
    ]
    rows = [
        # N2 industrial
        ["Nitrogen (N2)", "Industrial (99.9%)", "Nm3/h",
         "Incumbent: LIN delivered + tank", "Incumbent", 9,
         150000, 0.0, 0.22, 0.02, 5000,
         250.0, 0.70,
         "Fast deploy, minimal onsite equipment.",
         "Logistics + contract risk dominate total cost.",
         "Low utilization, temporary demand, or very small sites."],

        ["Nitrogen (N2)", "Industrial (99.9%)", "Nm3/h",
         "Platform: PSA N2 (99.9%)", "Platform", 9,
         450000, 0.40, 0.00, 0.05, 12000,
         250.0, 0.70,
         "Deletes deliveries; predictable unit cost at steady demand.",
         "Needs clean dry air + maintenance; purity tied to cycle + adsorbent health.",
         "Steady industrial demand (heat treat, metals, glass, packaging)."],

        ["Nitrogen (N2)", "Industrial (99.9%)", "Nm3/h",
         "Platform: Membrane N2 (95–99%)", "Platform", 9,
         300000, 0.30, 0.00, 0.04, 9000,
         250.0, 0.70,
         "Lower capex; simpler; fast install.",
         "Purity and recovery trade off; not ideal for tight specs.",
         "Smaller sites that can accept lower purity."],

        # O2 industrial
        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Incumbent: LOX delivered + tank", "Incumbent", 9,
         200000, 0.0, 0.30, 0.02, 5000,
         150.0, 0.70,
         "Simple ops: buy oxygen, consume oxygen.",
         "Logistics heavy; supply disruptions are real.",
         "Low/moderate use or highly variable demand."],

        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Platform: VPSA O2 (low pressure)", "Platform", 9,
         700000, 0.36, 0.00, 0.05, 18000,
         150.0, 0.70,
         "Strong economics when demand is steady; converts logistics spend into power + service.",
         "If customer needs higher pressure, booster adds cost + kWh.",
         "Glass/metals/wastewater enrichment at low pressure."],

        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Platform: VPSA O2 + booster (6 barg)", "Platform", 9,
         900000, 0.55, 0.00, 0.05, 20000,
         150.0, 0.70,
         "Same onsite supply + compatibility with higher-pressure distribution.",
         "Booster is a major cost/energy driver; avoid if process can accept low pressure.",
         "Sites needing pressurized O2 header."],

# Oxygen pressure/energy profiles (anchored to public refs)
O2_PROFILES = {
    "Low pressure VPSA (≈0.2 barg, no booster)": 0.36,   # Messer VPSA low specific power :contentReference[oaicite:10]{index=10}
    "Low pressure VSA (≈2.7 psig)": 0.39,               # PATH VSA ref :contentReference[oaicite:11]{index=11}
    "High pressure PSA (≈65 psig / ~4.5 barg)": 1.20,   # PATH PSA ref :contentReference[oaicite:12]{index=12}
}

if gas.startswith("Oxygen"):
    o2_profile = st.selectbox("O₂ delivery pressure class", list(O2_PROFILES.keys()))
    plat_kwh = O2_PROFILES[o2_profile]

    # Optional: if user chooses high pressure, warn them this is the expensive mode
    if "High pressure" in o2_profile:
        st.warning("High-pressure oxygen is expensive. Cheapest route is low-pressure O₂ + avoid boosters; use hybrid if peaks matter.")



        ["Oxygen (O2)", "Industrial (90–93%)", "Nm3/h",
         "Platform: Hybrid (VPSA base-load + LOX peaks)", "Platform", 9,
         650000, 0.36, 0.00, 0.05, 18000,
         150.0, 0.70,
         "Cheaper CAPEX vs full-size plant; resilience via LOX for peaks/maintenance windows.",
         "Still needs LOX contract/tank; economics depend on peak fraction.",
         "Customers who want uptime without oversizing."],

        # CO2
        ["Carbon Dioxide (CO2)", "Industrial", "t/yr",
         "Incumbent: LCO2 delivered + tank (industrial)", "Incumbent", 9,
         120000, 0.0, 350.0, 0.02, 4000,
         2000.0, 0.70,
         "Simplest when supply is stable.",
         "Shortage events can shut down production.",
         "Non-critical industrial uses."],

        ["Carbon Dioxide (CO2)", "Food/Bev", "t/yr",
         "Incumbent: LCO2 delivered + tank (food/bev)", "Incumbent", 9,
         120000, 0.0, 450.0, 0.02, 4000,
         2000.0, 0.70,
         "QA handled by supplier.",
         "Availability and pricing volatility.",
         "Food/bev where QA matters."],

        ["Carbon Dioxide (CO2)", "Industrial", "t/yr",
         "Platform: Conditioning + storage (no capture)", "Platform", 9,
         400000, 15.0, 0.00, 0.05, 12000,
         2000.0, 0.70,
         "You’re selling uptime: buffer + conditioning + telemetry + SLA.",
         "Does not generate CO2; still needs a source contract.",
         "Customers burned by outages."],
    ]
    return pd.DataFrame(rows, columns=cols)


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Gas Utility Platform TEA", page_icon="🧪", layout="wide")
inject_css()

tech = tech_library()

with st.sidebar:
    st.markdown("### 🧪 Gas Utility Platform")
    investor_mode = st.toggle("Investor Mode", value=True)
    st.caption("Investor Mode hides the spreadsheet vibe and keeps only the story + key controls.")

    st.divider()
    gas = st.selectbox("Gas", options=tech["gas"].unique().tolist(), index=0)
    purity_tier = st.selectbox("Purity tier", options=tech[tech["gas"] == gas]["purity_tier"].unique().tolist(), index=0)

    subset = tech[(tech["gas"] == gas) & (tech["purity_tier"] == purity_tier)].copy().reset_index(drop=True)
    unit = subset["unit"].iloc[0]

    st.divider()
    st.markdown("### Economics assumptions")
    electricity_price = st.number_input("Electricity price ($/kWh)", value=0.10, step=0.01, format="%.3f")
    wacc = st.number_input("WACC", value=0.12, step=0.01, format="%.3f")
    project_life = st.number_input("Project life (years)", value=10, step=1)
    capacity_factor = st.number_input("Capacity factor (flow-based)", value=0.90, step=0.01)
    contingency_pct = st.number_input("Contingency (fraction of CAPEX)", value=0.15, step=0.01)
    capex_scaling_exponent = st.number_input("CAPEX scaling exponent", value=0.70, step=0.05)

    # Hybrid control (only used when that option is selected)
    peak_frac = st.slider("Hybrid peak fraction (LOX share)", 0.0, 0.6, 0.2, 0.05)

# Demand input
st.markdown(f"<span class='pill'>{gas}</span><span class='pill'>{purity_tier}</span>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Requirement</div>", unsafe_allow_html=True)

default_base = float(subset["base_demand_value"].iloc[0])
demand = st.number_input("Demand (Nm³/h)" if unit == "Nm3/h" else "Demand (t/yr)", value=default_base, step=10.0 if unit == "Nm3/h" else 100.0)

annual_u = annual_units(demand, unit, capacity_factor)
unit_label = "$/Nm³" if unit == "Nm3/h" else "$/t"

# Option selection
inc_opts = subset[subset["category"] == "Incumbent"]["option"].tolist()
plat_opts = subset[subset["category"] == "Platform"]["option"].tolist()

c1, c2 = st.columns(2)
with c1:
    incumbent_choice = st.selectbox("Incumbent option", inc_opts, index=0)
with c2:
    platform_choice = st.selectbox("Platform option", plat_opts, index=0)

inc = subset[subset["option"] == incumbent_choice].iloc[0].to_dict()
plat = subset[subset["option"] == platform_choice].iloc[0].to_dict()

# Scale capex
inc_capex = scale_capex(float(inc["capex_installed"]), float(demand), float(inc["base_demand_value"]), capex_scaling_exponent)
plat_capex_raw = scale_capex(float(plat["capex_installed"]), float(demand), float(plat["base_demand_value"]), capex_scaling_exponent)

# Hybrid logic: size VPSA for base-load, buy LOX for peaks
# Only for O2 hybrid option.
plat_var_opex = float(plat["var_opex_per_unit"])
plat_kwh = float(plat["kwh_per_unit"])
plat_capex = plat_capex_raw

if platform_choice.startswith("Platform: Hybrid") and gas.startswith("Oxygen"):
    base_flow = max(0.0, (1.0 - peak_frac) * demand)
    # Resize CAPEX based on base_flow
    plat_capex = scale_capex(float(plat["capex_installed"]), float(base_flow), float(plat["base_demand_value"]), capex_scaling_exponent)
    # Add delivered LOX for the peak fraction (use incumbent delivered unit price as proxy)
    loX_price = float(inc["var_opex_per_unit"])
    plat_var_opex = peak_frac * loX_price  # $/Nm3 equivalent for peak supply blend

# Compute LCOG
inc_lcog, inc_br = lcog(
    annual_units_value=annual_u,
    capex_installed=inc_capex,
    wacc=wacc,
    project_life=int(project_life),
    contingency_pct=contingency_pct,
    fixed_om_pct=float(inc["fixed_om_pct"]),
    labor_per_year=float(inc["labor_per_year"]),
    electricity_price=electricity_price,
    kwh_per_unit=float(inc["kwh_per_unit"]),
    variable_opex_per_unit=float(inc["var_opex_per_unit"]),
)
plat_lcog, plat_br = lcog(
    annual_units_value=annual_u,
    capex_installed=plat_capex,
    wacc=wacc,
    project_life=int(project_life),
    contingency_pct=contingency_pct,
    fixed_om_pct=float(plat["fixed_om_pct"]),
    labor_per_year=float(plat["labor_per_year"]),
    electricity_price=electricity_price,
    kwh_per_unit=plat_kwh,
    variable_opex_per_unit=plat_var_opex,
)

inc_annual = inc_br.get("Total annual", float("nan"))
plat_annual = plat_br.get("Total annual", float("nan"))
annual_savings = inc_annual - plat_annual if (not math.isnan(inc_annual) and not math.isnan(plat_annual)) else float("nan")

inc_capex_total = inc_capex * (1 + contingency_pct)
plat_capex_total = plat_capex * (1 + contingency_pct)
delta_capex = plat_capex_total - inc_capex_total
payback = (delta_capex / annual_savings) if (annual_savings and annual_savings > 0 and delta_capex > 0) else float("nan")

# Power check (only meaningful for Nm3/h cases)
inc_kw = avg_power_kw(demand, float(inc["kwh_per_unit"])) if unit == "Nm3/h" else 0.0
plat_kw = avg_power_kw(demand, plat_kwh) if unit == "Nm3/h" else 0.0

# -----------------------------
# Tabs layout (clean + investor-friendly)
# -----------------------------
tab_overview, tab_econ, tab_process, tab_sens, tab_sources = st.tabs(
    ["Overview", "Economics", "Process flow", "Sensitivity", "Sources"]
)

with tab_overview:
    top1, top2, top3, top4 = st.columns(4)
    with top1:
        kpi("Annual units", f"{annual_u:,.0f}", "Nm³/yr" if unit == "Nm3/h" else "t/yr")
    with top2:
        kpi("Incumbent LCOG", "n/a" if math.isnan(inc_lcog) else f"${inc_lcog:,.3f}", unit_label)
    with top3:
        kpi("Platform LCOG", "n/a" if math.isnan(plat_lcog) else f"${plat_lcog:,.3f}", unit_label)
    with top4:
        if not math.isnan(inc_lcog) and not math.isnan(plat_lcog) and inc_lcog > 0:
            kpi("Savings", f"{(1 - plat_lcog / inc_lcog) * 100:,.1f}%", "vs incumbent")
        else:
            kpi("Savings", "n/a", "")

    mid1, mid2, mid3, mid4 = st.columns(4)
    with mid1:
        kpi("Inc CAPEX (+cont)", f"${inc_capex_total:,.0f}", "")
    with mid2:
        kpi("Plat CAPEX (+cont)", f"${plat_capex_total:,.0f}", "")
    with mid3:
        kpi("Annual savings", "n/a" if math.isnan(annual_savings) else f"${annual_savings:,.0f}/yr", "")
    with mid4:
        kpi("Payback", "n/a" if math.isnan(payback) else f"{payback:,.1f} yrs", "")

    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.markdown("**Why it can win**")
    st.write(f"• {plat['why']}")
    st.write(f"• Best fit: {plat['best_fit']}")
    st.write(f"• Watch-outs: {plat['watchouts']}")
    if gas.startswith("Oxygen") and "low pressure" in platform_choice.lower():
        st.write("• Cost lever: keep O₂ at low product pressure when process allows (avoid a large booster).")
    if gas.startswith("Oxygen") and "hybrid" in platform_choice.lower():
        st.write("• Hybrid lever: smaller VPSA for base-load + LOX for peaks reduces CAPEX while keeping uptime.")
    st.markdown("</div>", unsafe_allow_html=True)

    if unit == "Nm3/h":
        st.caption(f"Power sanity check: Incumbent ~{inc_kw:,.1f} kW, Platform ~{plat_kw:,.1f} kW (kWh/Nm³ × Nm³/h).")

with tab_econ:
    # Simple bar comparisons without extra libs
    df = pd.DataFrame([
        ["Incumbent", inc_lcog, inc_capex_total, inc_annual],
        ["Platform", plat_lcog, plat_capex_total, plat_annual],
    ], columns=["Scenario", "LCOG", "CAPEX_cont", "Annual_cost"])

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Cost breakdown (annual)**")
        br = pd.DataFrame([
            ["Annualized CAPEX", inc_br.get("Annualized CAPEX", 0), plat_br.get("Annualized CAPEX", 0)],
            ["Fixed O&M", inc_br.get("Fixed O&M", 0), plat_br.get("Fixed O&M", 0)],
            ["Labor", inc_br.get("Labor", 0), plat_br.get("Labor", 0)],
            ["Electricity", inc_br.get("Electricity", 0), plat_br.get("Electricity", 0)],
            ["Variable OPEX", inc_br.get("Variable OPEX", 0), plat_br.get("Variable OPEX", 0)],
        ], columns=["Item", "Incumbent $/yr", "Platform $/yr"])
        st.dataframe(br, use_container_width=True)

    with cB:
        st.markdown("**What drives the delta**")
        st.write("Incumbent is dominated by **delivered-gas unit price**. Platform shifts cost into **electricity + service**.")
        if gas.startswith("Oxygen"):
            st.write("For O₂, the biggest hidden lever is **product pressure**. Boosting pressure can dominate cost.")
        if gas.startswith("Carbon Dioxide"):
            st.write("CO₂ platform here is *not capture*. It’s a resilience + buffer product with telemetry/SLA.")

    st.markdown("**Payback quick view**")
    if not math.isnan(annual_savings) and annual_savings > 0 and delta_capex > 0:
        years = np.arange(0, int(project_life) + 1)
        cum = -delta_capex + annual_savings * years
        st.line_chart(pd.DataFrame({"Cumulative cashflow ($)": cum}, index=years))
    else:
        st.info("No payback curve: savings not positive or platform CAPEX not higher for this scenario.")

with tab_process:
    st.markdown("**Incumbent vs Platform flow**")

    if gas.startswith("Nitrogen"):
        render_flow(
            "Incumbent: delivered LIN",
            ["LIN delivery", "Bulk LIN tank", "Vaporizer", "Regulation", "N₂ header", "Point-of-use"],
            note="Simple ops. Cost + uptime tied to deliveries.",
            dashed=[(1, 4, "Buffer inventory")]
        )
        if "membrane" in platform_choice.lower():
            render_flow(
                "Platform: membrane N₂",
                ["Ambient air", "Compressor", "Dryer/filters", "Membrane", "Receiver", "N₂ header"],
                note="Lower capex; purity/recovery tradeoff.",
                dashed=[(3, 3, "O₂-rich permeate → vent")]
            )
        else:
            render_flow(
                "Platform: PSA N₂",
                ["Ambient air", "Compressor", "Dryer/filters", "PSA beds", "Receiver", "N₂ header"],
                note="Strong economics for steady industrial demand.",
                dashed=[(3, 3, "Offgas → vent")]
            )

    elif gas.startswith("Oxygen"):
        render_flow(
            "Incumbent: delivered LOX",
            ["LOX delivery", "Bulk LOX tank", "Vaporizer", "Regulation", "O₂ receiver", "O₂ header"],
            note="Simple. Logistics-heavy.",
            dashed=[(1, 4, "Buffer inventory")]
        )

        if "hybrid" in platform_choice.lower():
            render_flow(
                "Platform: hybrid (VPSA base + LOX peaks)",
                ["Ambient air", "VPSA unit", "O₂ receiver", "O₂ header", "LOX peak supply", "SLA monitoring"],
                note="Cheaper CAPEX vs oversizing. LOX covers peaks/maintenance.",
                dashed=[(4, 3, "Peak/backup feed")]
            )
        else:
            render_flow(
                "Platform: VPSA/VSA oxygen",
                ["Ambient air", "Blower/LP comp", "Filters", "VPSA beds + vacuum", "O₂ receiver", "O₂ header"],
                note="Most competitive at low product pressure.",
                dashed=[(3, 3, "N₂-rich tailgas → vent")]
            )

    else:
        render_flow(
            "Incumbent: delivered CO₂",
            ["LCO₂ delivery", "Bulk tank", "Pump/pressure build", "Vaporizer", "QA", "CO₂ header"],
            note="Works when supply is stable.",
            dashed=[(1, 5, "Buffer inventory")]
        )
        render_flow(
            "Platform: conditioning + storage (no capture)",
            ["Supply contract/source", "Bulk storage", "Conditioning", "Telemetry + QA", "CO₂ header", "SLA monitoring"],
            note="Resilience + SLA product (not CO₂ generation).",
            dashed=[(1, 4, "Extended buffer")]
        )

with tab_sens:
    st.markdown("**Sensitivity that VCs actually care about**")
    st.write("This shows how the decision flips as electricity price and delivered-gas price move.")

    if unit == "Nm3/h":
        # sweep electricity price
        ep = np.linspace(0.05, 0.25, 25)
        sweep = []
        for e in ep:
            _, br_p = lcog(annual_u, plat_capex, wacc, int(project_life), contingency_pct, float(plat["fixed_om_pct"]),
                           float(plat["labor_per_year"]), e, plat_kwh, plat_var_opex)
            _, br_i = lcog(annual_u, inc_capex, wacc, int(project_life), contingency_pct, float(inc["fixed_om_pct"]),
                           float(inc["labor_per_year"]), e, float(inc["kwh_per_unit"]), float(inc["var_opex_per_unit"]))
            sweep.append([e, br_i["Total annual"], br_p["Total annual"]])

        sdf = pd.DataFrame(sweep, columns=["Electricity ($/kWh)", "Incumbent $/yr", "Platform $/yr"]).set_index("Electricity ($/kWh)")
        st.line_chart(sdf)

    else:
        st.info("Sensitivity is most useful for flow-based gases. CO₂ is dominated by delivered contract terms + uptime value.")

with tab_sources:
    st.markdown("### Where the default energy numbers come from")
    st.write(
        "These are public references used to anchor default kWh/Nm³ values. "
        "Real quotes still vary with pressure, purity, site conditions, and vendor design."
    )
    st.write("• O₂ VPSA/VSA specific power examples: ~0.29–0.36 kWh/Nm³ reported by vendors and industry pages; ~0.39 kWh/Nm³ cited as a VSA reference in a PATH/CHAI brief.")
    st.write("• N₂ PSA specific power example: 0.3–0.6 kWh/Nm³ listed on a Messer nitrogen PSA info page.")
    st.write("• PSA/VSA oxygen capex references for healthcare settings (not industrial pricing) show order-of-magnitude costs and the PSA vs VSA premium.")

    st.markdown("**Links (for investors who ask)**")
    st.code(
        "\n".join([
            "Messer N2 PSA (specific power 0.3–0.6 kWh/Nm³): https://applications.messergroup.com/psa",
            "Messer VPSA O2 (specific power ~0.36 kWh/Nm³): https://applications.messergroup.com/oxygen_generator/vpsa",
            "PATH/CHAI oxygen generation brief (VSA ~0.39 kWh/Nm³ + capex examples): https://media.path.org/documents/O2_generation_and_storage_PSA_VSA_v1.pdf",
            "Vendor examples quoting VPSA ~0.29–0.32 kWh/Nm³: https://www.vpsatech.com/VPSA-Oxygen.html",
        ]),
        language="text",
    )
