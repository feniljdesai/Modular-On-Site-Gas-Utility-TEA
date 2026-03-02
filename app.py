import math
from typing import Dict, Tuple

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
# BOM interpolation helpers
# -----------------------------
def lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


def interpolate_bom(flow: float, anchors, components) -> pd.DataFrame:
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
    # N2 PSA
    n2_psa_components = {
        "Air compressor package (oil-free preferred)": {"Small": (20000, 60000), "Medium": (60000, 180000), "Large": (200000, 600000)},
        "Aftercooler + moisture separator + drains": {"Small": (2000, 8000), "Medium": (5000, 20000), "Large": (15000, 60000)},
        "Filtration train (particulate + coalescing)": {"Small": (1000, 5000), "Medium": (3000, 12000), "Large": (8000, 30000)},
        "Air dryer (refrigerated/desiccant) + dewpoint": {"Small": (5000, 25000), "Medium": (15000, 70000), "Large": (40000, 180000)},
        "PSA vessels + adsorbent (CMS)": {"Small": (15000, 60000), "Medium": (60000, 220000), "Large": (200000, 800000)},
        "Valve manifold + actuators": {"Small": (5000, 25000), "Medium": (20000, 80000), "Large": (60000, 250000)},
        "N2 buffer tank (receiver)": {"Small": (2000, 10000), "Medium": (8000, 35000), "Large": (25000, 100000)},
        "Analyzer package (O2 ppm, dewpoint optional)": {"Small": (3000, 15000), "Medium": (5000, 25000), "Large": (10000, 40000)},
        "PLC/HMI + remote telemetry": {"Small": (5000, 25000), "Medium": (10000, 40000), "Large": (20000, 80000)},
        "Skid piping, frame, wiring, assembly": {"Small": (10000, 40000), "Medium": (30000, 120000), "Large": (80000, 300000)},
        "Safety relief/vent routing": {"Small": (1000, 5000), "Medium": (3000, 12000), "Large": (8000, 30000)},
    }
    n2_psa_anchors = [(50.0, "Small"), (250.0, "Medium"), (1000.0, "Large")]

    # N2 Membrane
    n2_mem_components = {
        "Air compressor package": {"Small": (15000, 50000), "Medium": (50000, 150000), "Large": (180000, 550000)},
        "Aftercooler + moisture separator": {"Small": (2000, 8000), "Medium": (5000, 20000), "Large": (15000, 60000)},
        "Filtration + dryer (critical)": {"Small": (6000, 25000), "Medium": (15000, 60000), "Large": (40000, 180000)},
        "Membrane module(s)": {"Small": (8000, 40000), "Medium": (30000, 140000), "Large": (120000, 450000)},
        "N2 receiver + regulators": {"Small": (3000, 15000), "Medium": (10000, 40000), "Large": (30000, 120000)},
        "Analyzer + PLC/HMI": {"Small": (5000, 30000), "Medium": (12000, 50000), "Large": (25000, 90000)},
        "Skid integration": {"Small": (8000, 35000), "Medium": (25000, 100000), "Large": (70000, 250000)},
    }
    n2_mem_anchors = [(50.0, "Small"), (250.0, "Medium"), (1000.0, "Large")]

    # O2 VPSA
    o2_vpsa_components = {
        "Air blower / low-pressure compressor": {"Small": (15000, 60000), "Medium": (40000, 140000), "Large": (120000, 400000)},
        "Aftercooler + filtration train": {"Small": (4000, 20000), "Medium": (10000, 45000), "Large": (30000, 120000)},
        "Adsorber vessels + zeolite": {"Small": (25000, 120000), "Medium": (80000, 350000), "Large": (250000, 900000)},
        "Vacuum pump(s) + vacuum receiver": {"Small": (20000, 120000), "Medium": (70000, 300000), "Large": (200000, 900000)},
        "Valve manifold + actuators": {"Small": (8000, 40000), "Medium": (25000, 110000), "Large": (80000, 300000)},
        "O2 receiver tank + regulators": {"Small": (5000, 25000), "Medium": (12000, 60000), "Large": (35000, 160000)},
        "Analyzer (O2 %) + flow/pressure": {"Small": (4000, 18000), "Medium": (8000, 30000), "Large": (15000, 45000)},
        "PLC/HMI + telemetry": {"Small": (8000, 35000), "Medium": (15000, 55000), "Large": (25000, 90000)},
        "Oxygen-compatible piping/cleaning + skid": {"Small": (15000, 70000), "Medium": (40000, 160000), "Large": (120000, 450000)},
        "Safety relief/vent routing": {"Small": (2000, 10000), "Medium": (5000, 20000), "Large": (12000, 45000)},
    }
    o2_vpsa_anchors = [(50.0, "Small"), (150.0, "Medium"), (500.0, "Large")]

    # CO2 conditioning + storage (rough)
    co2_components = {
        "Bulk liquid CO2 storage tank(s) + foundations": (60000, 220000),
        "Vaporizer / heater + controls": (12000, 55000),
        "Transfer pump(s) / pressure builder": (8000, 45000),
        "Piping/valves/regulators + install": (20000, 90000),
        "Instrumentation (pressure/flow/temp) + QA": (8000, 45000),
        "PLC/HMI + telemetry": (8000, 45000),
        "Safety relief / venting / ODH signage": (5000, 25000),
    }

    return {
        "N2 PSA": (n2_psa_anchors, n2_psa_components),
        "N2 Membrane": (n2_mem_anchors, n2_mem_components),
        "O2 VPSA/VSA": (o2_vpsa_anchors, o2_vpsa_components),
        "CO2 Conditioning+Storage": co2_components,
    }


def bom_co2(demand_tpy: float, base_tpy: float = 2000.0, exponent: float = 0.70) -> pd.DataFrame:
    comps = bom_catalog()["CO2 Conditioning+Storage"]
    rows = []
    ratio = (demand_tpy / base_tpy) ** exponent if base_tpy > 0 and demand_tpy > 0 else 0.0
    for comp, (lo, hi) in comps.items():
        low = lo * ratio
        high = hi * ratio
        rows.append([comp, low, high, 0.5 * (low + high)])
    df = pd.DataFrame(rows, columns=["Component", "Low ($)", "High ($)", "Mid ($)"])
    total = df[["Low ($)", "High ($)", "Mid ($)"]].sum()
    df = pd.concat([df, pd.DataFrame([["TOTAL", total["Low ($)"], total["High ($)"], total["Mid ($)"]]], columns=df.columns)], ignore_index=True)
    return df


# -----------------------------
# Tech library (updated defaults)
# -----------------------------
def tech_library() -> pd.DataFrame:
    cols = [
        "gas",
        "option",
        "category",
        "trl",
        "capex_installed",
        "kwh_per_unit",
        "var_opex_per_unit",
        "fixed_om_pct",
        "labor_per_year",
        "notes",
        "base_demand_value",
        "base_demand_unit",
        "bom_family",
    ]
    rows = [
        # N2
        ["Nitrogen (N2)", "Incumbent: LIN delivered + tank", "Incumbent", 9, 150000, 0.0, 0.22, 0.02, 5000,
         "Bulk LIN delivery + onsite tank handling (price varies by region/contract).", 250.0, "Nm3/h", ""],
        ["Nitrogen (N2)", "Platform: On-site PSA (99.9%)", "Platform", 9, 450000, 0.28, 0.0, 0.05, 12000,
         "Industrial purity PSA; avoids trucking.", 250.0, "Nm3/h", "N2 PSA"],
        ["Nitrogen (N2)", "Platform: PSA + purifier (5N)", "Platform", 9, 800000, 0.42, 0.0, 0.05, 16000,
         "High purity; higher capex + kWh.", 250.0, "Nm3/h", "N2 PSA"],
        ["Nitrogen (N2)", "Platform: Membrane N2 (95–99%)", "Platform", 9, 300000, 0.25, 0.0, 0.04, 9000,
         "Lower capex; purity depends on staging.", 250.0, "Nm3/h", "N2 Membrane"],

        # O2
        ["Oxygen (O2)", "Incumbent: LOX delivered + tank", "Incumbent", 9, 200000, 0.0, 0.30, 0.02, 5000,
         "Bulk LOX delivery + tank; logistics-heavy.", 150.0, "Nm3/h", ""],
        ["Oxygen (O2)", "Platform: VPSA (90–95%)", "Platform", 9, 850000, 0.55, 0.0, 0.05, 18000,
         "Industrial oxygen; best for glass/metals/wastewater. Add LOX backup if needed.", 150.0, "Nm3/h", "O2 VPSA/VSA"],
        ["Oxygen (O2)", "Platform: micro-cryo ASU (99.5%+)", "Platform", 9, 3500000, 0.90, 0.0, 0.05, 40000,
         "High purity; higher capex.", 150.0, "Nm3/h", ""],

        # CO2
        ["Carbon Dioxide (CO2)", "Incumbent: Liquid CO2 delivered + tank (industrial)", "Incumbent", 9, 120000, 0.0, 350.0, 0.02, 4000,
         "Delivered LCO2 industrial grade (placeholder).", 2000.0, "t/yr", ""],
        ["Carbon Dioxide (CO2)", "Incumbent: Liquid CO2 delivered + tank (food/bev)", "Incumbent", 9, 120000, 0.0, 450.0, 0.02, 4000,
         "Delivered LCO2 food/bev (placeholder).", 2000.0, "t/yr", ""],
        ["Carbon Dioxide (CO2)", "Platform: Conditioning + storage (no capture)", "Platform", 9, 400000, 15.0, 0.0, 0.05, 12000,
         "Supply assurance + uptime SLA; does not create CO2.", 2000.0, "t/yr", "CO2 Conditioning+Storage"],

        # NH3 placeholder
        ["Ammonia (NH3)", "Incumbent: Delivered NH3 + tank", "Incumbent", 9, 250000, 0.0, 450.0, 0.02, 5000,
         "Optional / vector use-case dependent.", 1000.0, "t/yr", ""],
        ["Ammonia (NH3)", "Platform: NH3 storage + handling (optional)", "Platform", 6, 1200000, 0.0, 0.0, 0.05, 16000,
         "Optional resilience buffer.", 1000.0, "t/yr", ""],
    ]
    return pd.DataFrame(rows, columns=cols)


def default_loads() -> pd.DataFrame:
    return pd.DataFrame([
        {"gas": "Nitrogen (N2)", "avg_flow": 250.0, "unit": "Nm3/h", "annual_units": None},
        {"gas": "Oxygen (O2)", "avg_flow": 150.0, "unit": "Nm3/h", "annual_units": None},
        {"gas": "Carbon Dioxide (CO2)", "avg_flow": None, "unit": "t/yr", "annual_units": 2000.0},
        {"gas": "Ammonia (NH3)", "avg_flow": None, "unit": "t/yr", "annual_units": 0.0},
    ])


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Gas Utility TEA", layout="wide")
st.title("Modular On-Site Gas Utility TEA")
st.caption("Pick one gas, size the requirement, compare incumbent vs platform, and view customer economics + company economics.")

tech_df_base = tech_library()
loads_df = default_loads()

with st.sidebar:
    st.header("Select gas")
    gas = st.selectbox("Gas", options=loads_df["gas"].tolist(), index=0)

    st.divider()
    st.header("Global assumptions")
    electricity_price = st.number_input("Electricity price ($/kWh)", value=0.10, step=0.01, format="%.3f")
    wacc = st.number_input("WACC", value=0.12, step=0.01, format="%.3f")
    project_life = st.number_input("Project life (years)", value=10, step=1)
    capacity_factor = st.number_input("Capacity factor", value=0.90, step=0.01, format="%.3f")
    contingency_pct = st.number_input("Contingency (fraction of installed CAPEX)", value=0.15, step=0.01, format="%.3f")
    capex_scaling_exponent = st.number_input("CAPEX scaling exponent", value=0.70, step=0.05, format="%.2f")

    st.divider()
    st.header("Company view")
    integration_factor = st.number_input("Integration factor (BOM→build cost)", value=1.30, step=0.05, format="%.2f")
    install_factor = st.number_input("Install factor (build→installed)", value=1.25, step=0.05, format="%.2f")
    target_gross_margin = st.number_input("Target gross margin", value=0.35, step=0.05, format="%.2f")

# Requirement input for selected gas
lr = loads_df[loads_df["gas"] == gas].iloc[0].to_dict()
unit = lr["unit"]

st.subheader("1) Site requirement")
if unit == "Nm3/h":
    demand = st.number_input("Average flow (Nm³/h)", value=float(lr["avg_flow"] or 0.0), step=10.0)
else:
    demand = st.number_input("Annual demand (t/yr)", value=float(lr["annual_units"] or 0.0), step=100.0)

annual_units_value = annual_units(
    avg_flow=(demand if unit == "Nm3/h" else None),
    annual_units_value=(demand if unit == "t/yr" else None),
    unit=unit,
    capacity_factor=capacity_factor,
)

st.caption(
    f"Annual units computed using capacity factor = **{capacity_factor:.2f}** for flow-based gases. "
    f"Variable OPEX is **$/Nm³** for N₂/O₂ and **$/t** for CO₂/NH₃."
)

# Tech assumptions editor for selected gas
st.subheader("2) Technology assumptions (editable)")
gas_opts = tech_df_base[tech_df_base["gas"] == gas].copy().reset_index(drop=True)

with st.expander("Edit default CAPEX / kWh / delivered price (optional)", expanded=False):
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
        disabled=["gas", "category", "notes", "base_demand_unit", "bom_family"],
    )

inc_opts = gas_opts[gas_opts["category"] == "Incumbent"]["option"].tolist()
plat_opts = gas_opts[gas_opts["category"] == "Platform"]["option"].tolist()

st.subheader("3) Choose technologies")
c1, c2 = st.columns(2)
with c1:
    incumbent_choice = st.selectbox("Incumbent option", options=inc_opts, index=0)
with c2:
    platform_choice = st.selectbox("Platform option", options=plat_opts, index=0)

inc = gas_opts[gas_opts["option"] == incumbent_choice].iloc[0].to_dict()
plat = gas_opts[gas_opts["option"] == platform_choice].iloc[0].to_dict()

# Scale capex
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

st.subheader("4) Results (customer economics)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Annual units", f"{annual_units_value:,.0f} Nm³/yr" if unit == "Nm3/h" else f"{annual_units_value:,.0f} t/yr")
m2.metric("Incumbent LCOG", f"${inc_lcog:,.3f}/unit" if not math.isnan(inc_lcog) else "n/a")
m3.metric("Platform LCOG", f"${plat_lcog:,.3f}/unit" if not math.isnan(plat_lcog) else "n/a")
if not math.isnan(inc_lcog) and not math.isnan(plat_lcog) and inc_lcog > 0:
    m4.metric("Savings", f"{(1 - plat_lcog / inc_lcog) * 100:,.1f}%")
else:
    m4.metric("Savings", "n/a")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Incumbent installed CAPEX", f"${inc_capex_total:,.0f}")
m6.metric("Platform installed CAPEX", f"${plat_capex_total:,.0f}")
m7.metric("Annual savings", f"${annual_savings:,.0f}/yr" if not math.isnan(annual_savings) else "n/a")
m8.metric("Simple payback", f"{simple_payback:,.1f} yrs" if not math.isnan(simple_payback) else "n/a")

st.markdown("### Cost breakdown")
t1, t2 = st.columns(2)
with t1:
    st.markdown(f"**Incumbent: {incumbent_choice}**")
    if inc_br:
        st.dataframe(pd.DataFrame({"Cost item": list(inc_br.keys()), "Annual $": list(inc_br.values())}), use_container_width=True)
    else:
        st.info("No result (annual units is zero).")
with t2:
    st.markdown(f"**Platform: {platform_choice}**")
    if plat_br:
        st.dataframe(pd.DataFrame({"Cost item": list(plat_br.keys()), "Annual $": list(plat_br.values())}), use_container_width=True)
    else:
        st.info("No result (annual units is zero).")

if inc_br and plat_br:
    st.markdown("### Annual cost comparison")
    fig, ax = plt.subplots()
    labels = ["CAPEX", "Fixed O&M", "Labor", "Electricity", "Variable OPEX"]
    inc_vals = [inc_br.get("Annualized CAPEX", 0), inc_br.get("Fixed O&M", 0), inc_br.get("Labor", 0), inc_br.get("Electricity", 0), inc_br.get("Variable OPEX", 0)]
    plat_vals = [plat_br.get("Annualized CAPEX", 0), plat_br.get("Fixed O&M", 0), plat_br.get("Labor", 0), plat_br.get("Electricity", 0), plat_br.get("Variable OPEX", 0)]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, inc_vals, w, label="Incumbent")
    ax.bar(x + w / 2, plat_vals, w, label="Platform")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("$/yr")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

st.subheader("5) Sizing quick checks")
inc_kw = avg_power_kw((demand if unit == "Nm3/h" else None), (demand if unit == "t/yr" else None), unit, float(inc["kwh_per_unit"]))
plat_kw = avg_power_kw((demand if unit == "Nm3/h" else None), (demand if unit == "t/yr" else None), unit, float(plat["kwh_per_unit"]))
c3, c4 = st.columns(2)
c3.metric("Avg electric load (Incumbent)", f"{inc_kw:,.1f} kW")
c4.metric("Avg electric load (Platform)", f"{plat_kw:,.1f} kW")

st.subheader("6) Company view (BOM → build → installed → sale price)")
bom_family = str(plat.get("bom_family", "") or "")
bom_df = None

if gas == "Carbon Dioxide (CO2)" and bom_family == "CO2 Conditioning+Storage":
    bom_df = bom_co2(float(demand), base_tpy=float(plat["base_demand_value"]), exponent=capex_scaling_exponent)
elif bom_family in ("N2 PSA", "N2 Membrane", "O2 VPSA/VSA"):
    anchors, comps = bom_catalog()[bom_family]
    bom_df = interpolate_bom(float(demand), anchors, comps)

if bom_df is None:
    st.info("No BOM model attached to this platform option (yet).")
else:
    st.dataframe(bom_df, use_container_width=True)
    bom_mid = float(bom_df[bom_df["Component"] == "TOTAL"]["Mid ($)"].values[0])

    build_cost = bom_mid * integration_factor
    installed_cost_est = build_cost * install_factor
    installed_with_cont = installed_cost_est * (1 + contingency_pct)
    sale_price = installed_with_cont / max(1e-6, (1 - target_gross_margin))

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

st.subheader("7) Export")
out = pd.DataFrame([
    {"Gas": gas, "Scenario": "Incumbent", "Option": incumbent_choice, "Annual units": annual_units_value, "CAPEX installed ($)": inc_capex, "CAPEX+cont ($)": inc_capex_total,
     "LCOG ($/unit)": inc_lcog, "Annual cost ($/yr)": inc_annual, "Avg power (kW)": inc_kw},
    {"Gas": gas, "Scenario": "Platform", "Option": platform_choice, "Annual units": annual_units_value, "CAPEX installed ($)": plat_capex, "CAPEX+cont ($)": plat_capex_total,
     "LCOG ($/unit)": plat_lcog, "Annual cost ($/yr)": plat_annual, "Avg power (kW)": plat_kw},
])
st.download_button("Download selected_gas_results.csv", data=out.to_csv(index=False).encode("utf-8"),
                   file_name="selected_gas_results.csv", mime="text/csv")
