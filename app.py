import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Finance + TEA utilities
# -----------------------------
def crf(wacc: float, n_years: int) -> float:
    if n_years <= 0:
        raise ValueError("Project life must be > 0")
    if wacc == 0:
        return 1 / n_years
    return (wacc * (1 + wacc) ** n_years) / (((1 + wacc) ** n_years) - 1)


def annual_units(avg_flow, annual_units_value, unit: str, capacity_factor: float) -> float:
    if unit == "Nm3/h":
        return float(avg_flow or 0.0) * 8760.0 * capacity_factor
    if unit == "t/yr":
        return float(annual_units_value or 0.0)
    raise ValueError(f"Unsupported unit: {unit}")


def avg_power_kw(avg_flow, annual_units_value, unit: str, kwh_per_unit: float) -> float:
    if unit == "Nm3/h":
        return float(avg_flow or 0.0) * float(kwh_per_unit)
    if unit == "t/yr":
        return (float(annual_units_value or 0.0) * float(kwh_per_unit)) / 8760.0
    return 0.0


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
# Tech library (editable defaults)
# -----------------------------
def tech_library() -> pd.DataFrame:
    cols = [
        "gas", "purity_tier", "option", "category", "trl",
        "capex_installed", "kwh_per_unit", "var_opex_per_unit",
        "fixed_om_pct", "labor_per_year",
        "why_better", "limits", "best_fit",
        "base_demand_value", "base_demand_unit", "bom_family"
    ]
    rows = [
        # ---------------- N2 ----------------
        ["Nitrogen (N2)", "Industrial (99.9%)", "Incumbent: LIN delivered + tank", "Incumbent", 9,
         150000, 0.0, 0.22, 0.02, 5000,
         "Simple to deploy, zero onsite complexity. Good when utilization is low or demand is temporary.",
         "Cost is dominated by deliveries, contract terms, and logistics. Hard to guarantee price stability.",
         "Low utilization sites, short term projects, sites with tiny demand.",
         250.0, "Nm3/h", ""],

        ["Nitrogen (N2)", "Industrial (99.9%)", "Platform: On-site PSA (99.9%)", "Platform", 9,
         450000, 0.28, 0.00, 0.05, 12000,
         "Deletes trucking. Predictable cost. Good economics at steady demand.",
         "Needs compressor + maintenance. Not 5N purity without additional polishing.",
         "Steady industrial demand (heat treat, metals, glass, food packaging).",
         250.0, "Nm3/h", "N2 PSA"],

        ["Nitrogen (N2)", "Industrial (99.9%)", "Platform: Membrane N2 (95–99%)", "Platform", 9,
         300000, 0.25, 0.00, 0.04, 9000,
         "Lower capex than PSA. Fast install. Fewer valves.",
         "Purity and recovery tradeoff. Not ideal for tight specs.",
         "Smaller sites that can accept lower purity or variable purity.",
         250.0, "Nm3/h", "N2 Membrane"],

        ["Nitrogen (N2)", "High purity (5N)", "Incumbent: LIN delivered + tank", "Incumbent", 9,
         150000, 0.0, 0.28, 0.02, 5000,
         "Easiest path to high purity without onsite complexity.",
         "Expensive per unit. Logistics risk. Price volatility.",
         "Any site needing high purity without operators.",
         250.0, "Nm3/h", ""],

        ["Nitrogen (N2)", "High purity (5N)", "Platform: PSA + purifier (5N)", "Platform", 9,
         800000, 0.42, 0.00, 0.05, 16000,
         "High purity onsite with delivery elimination. Better when demand is steady and large.",
         "Higher capex and power. Purifier adds maintenance and consumables depending on design.",
         "Electronics adjacent, high spec inerting, specialty processing at steady loads.",
         250.0, "Nm3/h", "N2 PSA"],

        # ---------------- O2 ----------------
        ["Oxygen (O2)", "Industrial (90–95%)", "Incumbent: LOX delivered + tank", "Incumbent", 9,
         200000, 0.0, 0.30, 0.02, 5000,
         "Simple: you buy oxygen, you consume oxygen.",
         "Logistics-heavy. Uptime and supply depend on deliveries. Price is market + transport.",
         "Low/moderate use sites or highly variable demand.",
         150.0, "Nm3/h", ""],

        ["Oxygen (O2)", "Industrial (90–95%)", "Platform: VPSA (90–95%)", "Platform", 9,
         850000, 0.55, 0.00, 0.05, 18000,
         "Deletes trucking. Strong fit for steady industrial O2. Predictable operating cost.",
         "Higher capex. Power draw. Some sites still keep LOX tank for peak + resilience.",
         "Glass, metals, wastewater, cement enrichment with steady demand.",
         150.0, "Nm3/h", "O2 VPSA/VSA"],

        ["Oxygen (O2)", "High purity (99.5%+)", "Incumbent: LOX delivered + tank", "Incumbent", 9,
         200000, 0.0, 0.33, 0.02, 5000,
         "High purity supply without onsite plant.",
         "Same logistics risk. Higher delivered cost. Supply disruptions hurt production.",
         "Any high purity site without appetite for ASU ownership.",
         150.0, "Nm3/h", ""],

        ["Oxygen (O2)", "High purity (99.5%+)", "Platform: micro-cryo ASU (99.5%+)", "Platform", 9,
         3500000, 0.90, 0.00, 0.05, 40000,
         "Onsite high purity where it is mission-critical and volumes justify it.",
         "High capex and complexity. Usually only makes sense at larger demand or strict SLA.",
         "Sites where O2 is core feedstock and purity is non-negotiable.",
         150.0, "Nm3/h", ""],

        # ---------------- CO2 ----------------
        ["Carbon Dioxide (CO2)", "Industrial", "Incumbent: Liquid CO2 delivered + tank (industrial)", "Incumbent", 9,
         120000, 0.0, 350.0, 0.02, 4000,
         "Simple supply chain when CO2 is plentiful.",
         "In shortage events, CO2 can become unavailable. Uptime risk is real.",
         "Non-critical industrial uses where interruptions are tolerable.",
         2000.0, "t/yr", ""],

        ["Carbon Dioxide (CO2)", "Food/Bev", "Incumbent: Liquid CO2 delivered + tank (food/bev)", "Incumbent", 9,
         120000, 0.0, 450.0, 0.02, 4000,
         "Food-grade supply with QA handled by vendor.",
         "Availability and cost can swing. Logistics risk during peak demand seasons.",
         "Beverage/food users without onsite CO2 source.",
         2000.0, "t/yr", ""],

        ["Carbon Dioxide (CO2)", "Industrial", "Platform: Conditioning + storage (no capture)", "Platform", 9,
         400000, 15.0, 0.00, 0.05, 12000,
         "You’re buying resilience: buffer storage + conditioning + SLA. Less downtime.",
         "Does not generate CO2. Still needs supply source or contract.",
         "Customers burned by shortages who will pay for uptime.",
         2000.0, "t/yr", "CO2 Conditioning+Storage"],

        ["Carbon Dioxide (CO2)", "Food/Bev", "Platform: Conditioning + storage (no capture)", "Platform", 9,
         450000, 18.0, 0.00, 0.05, 14000,
         "Same story: uptime, buffering, QA-focused conditioning.",
         "Does not generate CO2. Must have reliable supplier or onsite stream.",
         "Food/bev sites where outages are catastrophic.",
         2000.0, "t/yr", "CO2 Conditioning+Storage"],
    ]
    return pd.DataFrame(rows, columns=cols)


def default_loads() -> pd.DataFrame:
    return pd.DataFrame([
        {"gas": "Nitrogen (N2)", "unit": "Nm3/h", "default": 250.0},
        {"gas": "Oxygen (O2)", "unit": "Nm3/h", "default": 150.0},
        {"gas": "Carbon Dioxide (CO2)", "unit": "t/yr", "default": 2000.0},
    ])


# -----------------------------
# BOM (simple, for "company view" pricing story)
# -----------------------------
def bom_co2(demand_tpy: float, base_tpy: float = 2000.0, exponent: float = 0.70) -> pd.DataFrame:
    comps = {
        "Bulk liquid CO2 storage tank(s) + foundations": (60000, 220000),
        "Vaporizer / heater + controls": (12000, 55000),
        "Transfer pump(s) / pressure builder": (8000, 45000),
        "Piping/valves/regulators + install": (20000, 90000),
        "Instrumentation + QA": (8000, 45000),
        "PLC/HMI + telemetry": (8000, 45000),
        "Safety relief / venting / ODH signage": (5000, 25000),
    }

    ratio = (demand_tpy / base_tpy) ** exponent if base_tpy > 0 and demand_tpy > 0 else 0.0
    rows = []
    for comp, (lo, hi) in comps.items():
        low = lo * ratio
        high = hi * ratio
        rows.append([comp, low, high, 0.5 * (low + high)])

    df = pd.DataFrame(rows, columns=["Component", "Low ($)", "High ($)", "Mid ($)"])
    total = df[["Low ($)", "High ($)", "Mid ($)"]].sum()
    df = pd.concat([df, pd.DataFrame([["TOTAL", total["Low ($)"], total["High ($)"], total["Mid ($)"]]], columns=df.columns)], ignore_index=True)
    return df


# -----------------------------
# Visualization helpers
# -----------------------------
COST_KEYS = ["Annualized CAPEX", "Fixed O&M", "Labor", "Electricity", "Variable OPEX"]


def stacked_cost_chart(inc_br: Dict[str, float], plat_br: Dict[str, float], title: str):
    labels = ["Incumbent", "Platform"]
    inc = np.array([inc_br.get(k, 0.0) for k in COST_KEYS])
    plat = np.array([plat_br.get(k, 0.0) for k in COST_KEYS])

    fig, ax = plt.subplots()
    bottom = np.zeros(2)

    for i, k in enumerate(COST_KEYS):
        vals = np.array([inc[i], plat[i]])
        ax.bar(labels, vals, bottom=bottom, label=k)
        bottom += vals

    ax.set_ylabel("Annual cost ($/yr)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    st.pyplot(fig, use_container_width=True)


def payback_cashflow_chart(delta_capex: float, annual_savings: float, years: int):
    # Cumulative cashflow: t=0 includes extra capex; then add annual savings each year
    xs = np.arange(0, years + 1)
    cum = np.zeros_like(xs, dtype=float)
    cum[0] = -delta_capex
    for t in range(1, years + 1):
        cum[t] = cum[t - 1] + annual_savings

    fig, ax = plt.subplots()
    ax.plot(xs, cum, marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative cashflow ($)")
    ax.set_title("Payback story: extra CAPEX vs cumulative savings")
    st.pyplot(fig, use_container_width=True)


def lcog_bar_chart(inc_lcog: float, plat_lcog: float, unit_label: str):
    fig, ax = plt.subplots()
    ax.bar(["Incumbent", "Platform"], [inc_lcog, plat_lcog])
    ax.set_ylabel(f"LCOG ({unit_label})")
    ax.set_title("Levelized cost comparison")
    st.pyplot(fig, use_container_width=True)

# -----------------------------
# Process flow diagram (matplotlib, no extra deps)
# -----------------------------
def _draw_flow(ax, blocks, title, branch=None, note=None):
    """
    blocks: list[str] laid left->right
    branch: list[dict] optional extra arrows like:
        {"from": int, "to": int, "text": str, "style": "dashed"}
    """
    ax.set_axis_off()
    ax.set_title(title, pad=12)

    n = len(blocks)
    xs = np.linspace(0.08, 0.92, n)
    y = 0.55

    # draw boxes
    for i, (x, label) in enumerate(zip(xs, blocks)):
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.2),
            transform=ax.transAxes,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.055, y),
                xytext=(x + 0.055, y),
                arrowprops=dict(arrowstyle="->", lw=1.6),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )

    # branches (backup / vent / offgas)
    if branch:
        for b in branch:
            i0, i1 = b["from"], b["to"]
            text = b.get("text", "")
            style = b.get("style", "solid")
            y0 = b.get("y0", 0.25)
            y1 = b.get("y1", 0.55)

            arrowprops = dict(arrowstyle="->", lw=1.4)
            if style == "dashed":
                arrowprops["linestyle"] = "--"

            ax.annotate(
                "",
                xy=(xs[i1], y1),
                xytext=(xs[i0], y0),
                arrowprops=arrowprops,
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
            if text:
                ax.text(
                    (xs[i0] + xs[i1]) / 2,
                    (y0 + y1) / 2 + 0.05,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    transform=ax.transAxes,
                )

    if note:
        ax.text(0.02, 0.05, note, fontsize=9, transform=ax.transAxes)


def flow_spec(gas: str, option: str, category: str):
    g = gas.lower()
    o = option.lower()

    # N2
    if "nitrogen" in g:
        if "delivered" in o:
            blocks = ["LIN truck", "Bulk LIN tank", "Vaporizer", "Pressure regulation", "N₂ header", "Point-of-use"]
            branch = [{"from": 1, "to": 4, "text": "Peak buffer", "style": "dashed", "y0": 0.25, "y1": 0.55}]
            note = "Delivered model: cost + uptime tied to logistics and contracts."
            return blocks, branch, note

        if "membrane" in o:
            blocks = ["Ambient air", "Compressor", "Filters + dryer", "Membrane module", "N₂ receiver", "N₂ header"]
            branch = [{"from": 3, "to": 3, "text": "Permeate/O₂-rich offgas → vent", "style": "dashed", "y0": 0.25, "y1": 0.55}]
            note = "Membrane: lower CAPEX, purity/recovery tradeoff."
            return blocks, branch, note

        # PSA / PSA+purifier
        blocks = ["Ambient air", "Compressor", "Filters + dryer", "PSA beds", "N₂ receiver", "N₂ header"]
        branch = [{"from": 3, "to": 3, "text": "Offgas → vent", "style": "dashed", "y0": 0.25, "y1": 0.55}]
        if "purifier" in o or "5n" in o:
            blocks = ["Ambient air", "Compressor", "Filters + dryer", "PSA beds", "Purifier", "N₂ receiver", "N₂ header"]
            branch = [{"from": 3, "to": 3, "text": "Offgas → vent", "style": "dashed", "y0": 0.25, "y1": 0.55}]
            note = "PSA+purifier: higher purity, higher CAPEX + power."
            return blocks, branch, note
        note = "PSA: strongest economics at steady industrial demand."
        return blocks, branch, note

    # O2
    if "oxygen" in g:
        if "delivered" in o:
            blocks = ["LOX truck", "Bulk LOX tank", "Vaporizer", "Pressure regulation", "O₂ receiver", "O₂ header"]
            branch = [{"from": 1, "to": 4, "text": "Peak buffer", "style": "dashed", "y0": 0.25, "y1": 0.55}]
            note = "Delivered model: logistics-heavy; simple ops."
            return blocks, branch, note

        if "vpsa" in o or "vsa" in o:
            blocks = ["Ambient air", "Blower/LP comp", "Filters", "VPSA beds + vacuum", "O₂ receiver", "O₂ header"]
            branch = [
                {"from": 3, "to": 3, "text": "N₂-rich tailgas → vent", "style": "dashed", "y0": 0.25, "y1": 0.55},
                # optional backup path shown conceptually
                {"from": 0, "to": 4, "text": "Optional LOX backup → receiver", "style": "dashed", "y0": 0.25, "y1": 0.55},
            ]
            note = "VPSA: best for 90–95% O₂; consider LOX backup only for peaks/resilience."
            return blocks, branch, note

        if "cryo" in o or "asu" in o:
            blocks = ["Ambient air", "Compressor", "Pre-purification", "Cold box", "O₂ product tank", "O₂ header"]
            branch = [{"from": 3, "to": 3, "text": "N₂/Ar coproducts (optional)", "style": "dashed", "y0": 0.25, "y1": 0.55}]
            note = "Micro-cryo ASU: high purity; high CAPEX + complexity."
            return blocks, branch, note

    # CO2
    if "carbon dioxide" in g or "co2" in g:
        if "delivered" in o:
            blocks = ["Liquid CO₂ truck", "Bulk LCO₂ tank", "Pump/pressure build", "Vaporizer", "Quality check", "CO₂ header"]
            branch = [{"from": 1, "to": 5, "text": "Buffer inventory", "style": "dashed", "y0": 0.25, "y1": 0.55}]
            note = "Delivered CO₂: simple when supply is stable; vulnerable to shortage events."
            return blocks, branch, note

        # conditioning + storage
        blocks = ["Supply contract / source", "Bulk LCO₂ storage", "Conditioning (pump/vaporizer)", "QA + telemetry", "CO₂ header", "SLA monitoring"]
        branch = [{"from": 1, "to": 4, "text": "Extended buffer", "style": "dashed", "y0": 0.25, "y1": 0.55}]
        note = "Platform CO₂ (no capture): selling uptime + buffering + monitoring (not CO₂ generation)."
        return blocks, branch, note

    # default fallback
    blocks = ["Input", "Process", "Output"]
    return blocks, None, "Flow not defined for this selection yet."
# -----------------------------
# "Why better" narrative builder
# -----------------------------
def decision_notes(gas: str, purity_tier: str, unit: str, demand: float, inc: dict, plat: dict,
                   inc_br: dict, plat_br: dict, inc_lcog: float, plat_lcog: float, savings: float) -> Dict[str, List[str]]:
    pros = []
    cons = []
    why = []

    # Always include curated text first
    why.append(str(plat.get("why_better", "")).strip())
    cons.append(str(plat.get("limits", "")).strip())
    pros.append(str(plat.get("best_fit", "")).strip())

    # Add computed drivers
    if not math.isnan(inc_lcog) and not math.isnan(plat_lcog):
        if plat_lcog < inc_lcog:
            why.append("Lower cost comes mainly from avoiding delivered-gas logistics and stabilizing OPEX.")
        else:
            why.append("Platform is not cheaper at this demand/purity; it may still win on uptime or control.")

    # Power story
    if float(plat.get("kwh_per_unit", 0.0)) > 0:
        why.append("Tradeoff: you convert some logistics cost into electricity + maintenance (more controllable).")

    # Demand heuristics (very lightweight)
    if gas.startswith("Nitrogen") and unit == "Nm3/h":
        if demand >= 150:
            why.append("At steady demand, onsite generation typically starts to win because deliveries become the dominant cost.")
        else:
            why.append("At lower demand, delivery can stay competitive unless you value uptime and price stability.")
    if gas.startswith("Oxygen") and unit == "Nm3/h":
        if "VPSA" in str(plat.get("option", "")) and demand >= 100:
            why.append("VPSA is a strong fit for 90–95% O2 when demand is steady; add LOX backup only if peaks/resilience require it.")
    if gas.startswith("Carbon Dioxide"):
        why.append("CO2 platform option here is resilience-focused: buffer + conditioning + SLA, not CO2 generation.")

    # Savings callout
    if not math.isnan(savings):
        if savings > 0:
            why.append("This configuration reduces annual total cost while improving onsite control.")
        else:
            why.append("This configuration increases annual cost; only justify it for uptime/SLA or strategic reasons.")

    # Clean empties
    pros = [x for x in pros if x]
    cons = [x for x in cons if x]
    why = [x for x in why if x]

    return {"why": why, "best_fit": pros, "limits": cons}


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Gas Utility TEA", layout="wide")

st.title("Modular On-Site Gas Utility TEA")
st.caption("One gas at a time. Story-driven visuals for customers, partners, and investors.")

tech_df = tech_library()
loads_df = default_loads()

with st.sidebar:
    st.header("1) Select gas + purity")
    gas = st.selectbox("Gas", options=loads_df["gas"].tolist(), index=0)

    purity_options = tech_df[tech_df["gas"] == gas]["purity_tier"].unique().tolist()
    purity_tier = st.selectbox("Purity tier", options=purity_options, index=0)

    st.divider()
    st.header("2) Global assumptions")
    electricity_price = st.number_input("Electricity price ($/kWh)", value=0.10, step=0.01, format="%.3f")
    wacc = st.number_input("WACC", value=0.12, step=0.01, format="%.3f")
    project_life = st.number_input("Project life (years)", value=10, step=1)
    capacity_factor = st.number_input("Capacity factor (flow-based)", value=0.90, step=0.01, format="%.3f")
    contingency_pct = st.number_input("Contingency (fraction of CAPEX)", value=0.15, step=0.01, format="%.3f")
    capex_scaling_exponent = st.number_input("CAPEX scaling exponent", value=0.70, step=0.05, format="%.2f")

    st.divider()
    st.header("3) Company view (pricing story)")
    integration_factor = st.number_input("Integration factor (BOM→build)", value=1.30, step=0.05, format="%.2f")
    install_factor = st.number_input("Install factor (build→installed)", value=1.25, step=0.05, format="%.2f")
    target_gm = st.number_input("Target gross margin", value=0.35, step=0.05, format="%.2f")

# Demand input
lr = loads_df[loads_df["gas"] == gas].iloc[0].to_dict()
unit = lr["unit"]
default_demand = float(lr["default"])

st.subheader("A) Requirement")
if unit == "Nm3/h":
    demand = st.number_input("Average flow (Nm³/h)", value=default_demand, step=10.0)
else:
    demand = st.number_input("Annual demand (t/yr)", value=default_demand, step=100.0)

annual_units_value = annual_units(
    avg_flow=(demand if unit == "Nm3/h" else None),
    annual_units_value=(demand if unit == "t/yr" else None),
    unit=unit,
    capacity_factor=capacity_factor,
)

unit_label = "$/Nm³" if unit == "Nm3/h" else "$/t"

st.caption(
    f"Annual units computed as **{annual_units_value:,.0f}** "
    f"{'Nm³/yr' if unit=='Nm3/h' else 't/yr'}. "
    f"LCOG output units: **{unit_label}**."
)

# Filter tech options for gas + purity tier
gas_opts = tech_df[(tech_df["gas"] == gas) & (tech_df["purity_tier"] == purity_tier)].copy().reset_index(drop=True)

# Let user tweak assumptions (but keep it optional)
st.subheader("B) Technology assumptions (optional edit)")
with st.expander("Edit CAPEX / kWh / delivered price (for scenario tuning)", expanded=False):
    gas_opts = st.data_editor(
        gas_opts,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "capex_installed": st.column_config.NumberColumn("CAPEX installed ($)", format="%.0f"),
            "kwh_per_unit": st.column_config.NumberColumn("kWh per unit", format="%.3f"),
            "var_opex_per_unit": st.column_config.NumberColumn("Variable OPEX ($/unit)", format="%.3f"),
            "fixed_om_pct": st.column_config.NumberColumn("Fixed O&M (fraction/yr)", format="%.3f"),
            "labor_per_year": st.column_config.NumberColumn("Labor ($/yr)", format="%.0f"),
        },
        disabled=["gas", "purity_tier", "category", "why_better", "limits", "best_fit", "base_demand_unit", "bom_family"],
    )

inc_opts = gas_opts[gas_opts["category"] == "Incumbent"]["option"].tolist()
plat_opts = gas_opts[gas_opts["category"] == "Platform"]["option"].tolist()

st.subheader("C) Choose comparison")
c1, c2 = st.columns(2)
with c1:
    incumbent_choice = st.selectbox("Incumbent option", options=inc_opts, index=0)
with c2:
    platform_choice = st.selectbox("Platform option", options=plat_opts, index=0)

inc = gas_opts[gas_opts["option"] == incumbent_choice].iloc[0].to_dict()
plat = gas_opts[gas_opts["option"] == platform_choice].iloc[0].to_dict()

# Scale CAPEX for demand
inc_capex = scale_capex(float(inc["capex_installed"]), float(demand), float(inc["base_demand_value"]), capex_scaling_exponent) if demand > 0 else 0.0
plat_capex = scale_capex(float(plat["capex_installed"]), float(demand), float(plat["base_demand_value"]), capex_scaling_exponent) if demand > 0 else 0.0

# TEA
inc_lcog, inc_br = lcog(
    annual_units_value=annual_units_value,
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
    annual_units_value=annual_units_value,
    capex_installed=plat_capex,
    wacc=wacc,
    project_life=int(project_life),
    contingency_pct=contingency_pct,
    fixed_om_pct=float(plat["fixed_om_pct"]),
    labor_per_year=float(plat["labor_per_year"]),
    electricity_price=electricity_price,
    kwh_per_unit=float(plat["kwh_per_unit"]),
    variable_opex_per_unit=float(plat["var_opex_per_unit"]),
)

inc_annual = inc_br.get("Total annual", float("nan"))
plat_annual = plat_br.get("Total annual", float("nan"))
annual_savings = inc_annual - plat_annual if (not math.isnan(inc_annual) and not math.isnan(plat_annual)) else float("nan")

inc_capex_total = inc_capex * (1 + contingency_pct)
plat_capex_total = plat_capex * (1 + contingency_pct)
delta_capex = plat_capex_total - inc_capex_total

simple_payback = (delta_capex / annual_savings) if (annual_savings and annual_savings > 0 and delta_capex > 0) else float("nan")

inc_kw = avg_power_kw((demand if unit == "Nm3/h" else None), (demand if unit == "t/yr" else None), unit, float(inc["kwh_per_unit"]))
plat_kw = avg_power_kw((demand if unit == "Nm3/h" else None), (demand if unit == "t/yr" else None), unit, float(plat["kwh_per_unit"]))

# -----------------------------
# Executive summary (top)
# -----------------------------
st.subheader("D) Executive summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Annual units", f"{annual_units_value:,.0f} Nm³/yr" if unit == "Nm3/h" else f"{annual_units_value:,.0f} t/yr")
m2.metric("Incumbent LCOG", f"${inc_lcog:,.3f} {unit_label}" if not math.isnan(inc_lcog) else "n/a")
m3.metric("Platform LCOG", f"${plat_lcog:,.3f} {unit_label}" if not math.isnan(plat_lcog) else "n/a")
if not math.isnan(inc_lcog) and not math.isnan(plat_lcog) and inc_lcog > 0:
    m4.metric("Savings", f"{(1 - plat_lcog / inc_lcog) * 100:,.1f}%")
else:
    m4.metric("Savings", "n/a")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Incumbent CAPEX (with contingency)", f"${inc_capex_total:,.0f}")
m6.metric("Platform CAPEX (with contingency)", f"${plat_capex_total:,.0f}")
m7.metric("Annual savings", f"${annual_savings:,.0f}/yr" if not math.isnan(annual_savings) else "n/a")
m8.metric("Simple payback", f"{simple_payback:,.1f} yrs" if not math.isnan(simple_payback) else "n/a")

st.caption("Interpretation: platform often trades **higher CAPEX** for **lower ongoing delivered-gas cost** and better control. For CO₂, the platform option is **uptime / buffer / SLA** oriented.")

# -----------------------------
# Narrative: why this wins
# -----------------------------
st.subheader("E) Why the platform can be better")
notes = decision_notes(gas, purity_tier, unit, float(demand), inc, plat, inc_br, plat_br, inc_lcog, plat_lcog, annual_savings)

cA, cB, cC = st.columns(3)
with cA:
    st.markdown("**Why it wins**")
    for x in notes["why"]:
        st.write(f"• {x}")
with cB:
    st.markdown("**Best fit**")
    for x in notes["best_fit"]:
        st.write(f"• {x}")
with cC:
    st.markdown("**Limits / watch-outs**")
    for x in notes["limits"]:
        st.write(f"• {x}")

# -----------------------------
# Visuals
# -----------------------------
st.subheader("F) Visual comparison")

v1, v2 = st.columns([1, 1])
with v1:
    if not math.isnan(inc_lcog) and not math.isnan(plat_lcog):
        lcog_bar_chart(inc_lcog, plat_lcog, unit_label)
    else:
        st.info("No LCOG plot (requirement is zero).")

with v2:
    if inc_br and plat_br:
        stacked_cost_chart(inc_br, plat_br, "Annual cost breakdown (stacked)")
    else:
        st.info("No annual cost breakdown (requirement is zero).")

st.subheader("G) Payback visualization")
if not math.isnan(annual_savings) and annual_savings > 0 and delta_capex > 0:
    payback_cashflow_chart(delta_capex=delta_capex, annual_savings=annual_savings, years=int(project_life))
else:
    st.info("Payback curve not shown because either (a) platform CAPEX is not higher, or (b) savings are not positive at this configuration.")

st.subheader("H) Power + operating reality")
p1, p2, p3 = st.columns(3)
p1.metric("Avg power (Incumbent)", f"{inc_kw:,.1f} kW")
p2.metric("Avg power (Platform)", f"{plat_kw:,.1f} kW")
p3.metric("Platform power", f"{plat_kw/1000:,.3f} MW")

st.caption("Power draw is what investors and customers usually miss. The platform wins when electricity is stable and delivered gas is expensive or unreliable.")

st.subheader("Process flow diagram")

view_mode = st.radio(
    "Show flow for",
    options=["Both", "Incumbent only", "Platform only"],
    horizontal=True
)

if view_mode in ("Both", "Incumbent only"):
    inc_blocks, inc_branch, inc_note = flow_spec(gas, incumbent_choice, "Incumbent")
if view_mode in ("Both", "Platform only"):
    plat_blocks, plat_branch, plat_note = flow_spec(gas, platform_choice, "Platform")

if view_mode == "Both":
    f1, f2 = st.columns(2)
    with f1:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        _draw_flow(ax, inc_blocks, "Incumbent process flow", branch=inc_branch, note=inc_note)
        st.pyplot(fig, use_container_width=True)
    with f2:
        fig, ax = plt.subplots(figsize=(8, 2.4))
        _draw_flow(ax, plat_blocks, "Platform process flow", branch=plat_branch, note=plat_note)
        st.pyplot(fig, use_container_width=True)

elif view_mode == "Incumbent only":
    fig, ax = plt.subplots(figsize=(10, 2.6))
    _draw_flow(ax, inc_blocks, "Incumbent process flow", branch=inc_branch, note=inc_note)
    st.pyplot(fig, use_container_width=True)

else:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    _draw_flow(ax, plat_blocks, "Platform process flow", branch=plat_branch, note=plat_note)
    st.pyplot(fig, use_container_width=True)

# -----------------------------
# Company view: BOM / pricing story
# -----------------------------
st.subheader("I) Company view (what does it cost us?)")

bom_df = None
if gas.startswith("Carbon Dioxide") and str(plat.get("bom_family", "")) == "CO2 Conditioning+Storage":
    bom_df = bom_co2(float(demand), base_tpy=float(plat["base_demand_value"]), exponent=float(capex_scaling_exponent))

if bom_df is None:
    st.info("BOM model in this demo is implemented for CO₂ conditioning + storage. (We can add N₂ PSA / O₂ VPSA BOM next.)")
else:
    st.dataframe(bom_df, use_container_width=True)
    bom_mid = float(bom_df[bom_df["Component"] == "TOTAL"]["Mid ($)"].values[0])

    build_cost = bom_mid * integration_factor
    installed_cost_est = build_cost * install_factor
    installed_with_cont = installed_cost_est * (1 + contingency_pct)
    sale_price = installed_with_cont / max(1e-6, (1 - target_gm))

    company = pd.DataFrame(
        [
            ["BOM mid (equipment)", bom_mid],
            ["Build cost (BOM × integration)", build_cost],
            ["Installed cost (build × install)", installed_cost_est],
            ["Installed + contingency", installed_with_cont],
            ["Suggested sale price (target GM)", sale_price],
        ],
        columns=["Item", "$"],
    )
    st.dataframe(company, use_container_width=True)

# -----------------------------
# Exports
# -----------------------------
st.subheader("J) Export")
summary = pd.DataFrame([
    {
        "Gas": gas,
        "Purity tier": purity_tier,
        "Scenario": "Incumbent",
        "Option": incumbent_choice,
        "Demand": demand,
        "Demand unit": unit,
        "Annual units": annual_units_value,
        "CAPEX installed ($)": inc_capex,
        "CAPEX+cont ($)": inc_capex_total,
        "LCOG": inc_lcog,
        "Annual cost ($/yr)": inc_annual,
        "Avg power (kW)": inc_kw,
        "TRL": int(inc["trl"]),
    },
    {
        "Gas": gas,
        "Purity tier": purity_tier,
        "Scenario": "Platform",
        "Option": platform_choice,
        "Demand": demand,
        "Demand unit": unit,
        "Annual units": annual_units_value,
        "CAPEX installed ($)": plat_capex,
        "CAPEX+cont ($)": plat_capex_total,
        "LCOG": plat_lcog,
        "Annual cost ($/yr)": plat_annual,
        "Avg power (kW)": plat_kw,
        "TRL": int(plat["trl"]),
    },
])

st.download_button(
    "Download results.csv",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="gas_tea_results.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("Next upgrade: add N₂ PSA + O₂ VPSA BOM models and a 'Recommended option' selector based on purity + demand + payback threshold.")
