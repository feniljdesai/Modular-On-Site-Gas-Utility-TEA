import math
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

def annual_units_from_load(load_row: dict, capacity_factor: float) -> float:
    unit = load_row["unit"]
    if unit == "Nm3/h":
        return float(load_row["avg_flow"]) * 8760.0 * capacity_factor
    if unit == "t/yr":
        return float(load_row["annual_units"])
    raise ValueError(f"Unsupported unit: {unit}")

def avg_power_kw(load_row: dict, kwh_per_unit: float) -> float:
    unit = load_row["unit"]
    if unit == "Nm3/h":
        return float(load_row["avg_flow"]) * float(kwh_per_unit)
    if unit == "t/yr":
        return (float(load_row["annual_units"]) * float(kwh_per_unit)) / 8760.0
    return 0.0

def scale_capex(base_capex: float, demand: float, base_demand: float, exponent: float) -> float:
    if base_demand <= 0:
        return base_capex
    if demand <= 0:
        return 0.0
    return base_capex * (demand / base_demand) ** exponent

def lcog(
    annual_units: float,
    capex_installed: float,
    wacc: float,
    project_life: int,
    contingency_pct: float,
    fixed_om_pct: float,
    labor_per_year: float,
    electricity_price: float,
    kwh_per_unit: float,
    variable_opex_per_unit: float,
    include_shared_labor: bool,
    shared_labor_pool: float,
    shared_labor_split: float,
):
    if annual_units <= 0:
        return float("nan"), {}

    capex_total = capex_installed * (1.0 + contingency_pct)
    annualized_capex = capex_total * crf(wacc, project_life)
    fixed_om = capex_total * fixed_om_pct
    electricity = annual_units * kwh_per_unit * electricity_price
    variable_opex = annual_units * variable_opex_per_unit
    shared_labor = (shared_labor_pool * shared_labor_split) if include_shared_labor else 0.0

    total_annual = annualized_capex + fixed_om + labor_per_year + electricity + variable_opex + shared_labor
    breakdown = {
        "Annualized CAPEX": annualized_capex,
        "Fixed O&M": fixed_om,
        "Labor": labor_per_year,
        "Electricity": electricity,
        "Variable OPEX": variable_opex,
        "Shared labor": shared_labor,
        "Total annual": total_annual,
    }
    return total_annual / annual_units, breakdown

# -----------------------------
# Defaults (from your table)
# -----------------------------
def default_assumptions():
    return {
        "electricity_price": 0.10,
        "wacc": 0.12,
        "project_life": 10,
        "capacity_factor": 0.90,
        "contingency_pct": 0.15,
        "shared_labor_pool": 60000.0,
        "capex_scaling_exponent": 0.70,
        # Keep OFF to match your pasted LCOG examples which didn’t allocate shared labor per gas
        "include_shared_labor_in_lcog": False,
    }

def default_loads():
    return pd.DataFrame([
        {"gas": "Nitrogen (N2)", "avg_flow": 250.0, "unit": "Nm3/h", "annual_units": None, "backup_tanks": True, "purity_case": "Industrial (99.9%) + High (5N)"},
        {"gas": "Oxygen (O2)", "avg_flow": 150.0, "unit": "Nm3/h", "annual_units": None, "backup_tanks": True, "purity_case": "Industrial (90–95%) + High (99.5%+)"},
        {"gas": "Carbon Dioxide (CO2)", "avg_flow": None, "unit": "t/yr", "annual_units": 2000.0, "backup_tanks": "Optional", "purity_case": "Industrial (food/bev)"},
        {"gas": "Ammonia (NH3)", "avg_flow": None, "unit": "t/yr", "annual_units": 0.0, "backup_tanks": "Optional", "purity_case": "Buffer / import vector"},
    ])

def default_tech_library():
    cols = ["gas","option","category","trl","capex_installed","kwh_per_unit","var_opex_per_unit","fixed_om_pct","labor_per_year","notes","base_demand_value","base_demand_unit"]
    rows = [
        ["Nitrogen (N2)","Incumbent: LIN delivered + tank","Incumbent",9,200000,0.0,0.32,0.02,5000,"Delivered LIN baseline; includes tank rental/handling",250.0,"Nm3/h"],
        ["Nitrogen (N2)","Platform: On-site PSA (99.9%)","Platform",9,500000,0.30,0.0,0.05,15000,"Deletes deliveries; OPEX mainly power + service",250.0,"Nm3/h"],
        ["Nitrogen (N2)","Platform: PSA + purifier (5N)","Platform",9,900000,0.45,0.0,0.05,20000,"High purity via purifier; higher kWh and CAPEX",250.0,"Nm3/h"],

        ["Oxygen (O2)","Incumbent: LOX delivered + tank","Incumbent",9,250000,0.0,0.33,0.02,5000,"Delivered LOX baseline; trucks + storage",150.0,"Nm3/h"],
        ["Oxygen (O2)","Platform: VPSA (90–95%)","Platform",9,700000,0.45,0.0,0.05,15000,"Best fit for industrial purity; LOX backup for peaks",150.0,"Nm3/h"],
        ["Oxygen (O2)","Platform: micro-cryo ASU (99.5%+)","Platform",9,3000000,0.80,0.0,0.05,30000,"High purity is purity-driven; cost higher but reliable",150.0,"Nm3/h"],

        ["Carbon Dioxide (CO2)","Incumbent: Liquid CO2 delivered + tank","Incumbent",9,150000,0.0,590.0,0.02,5000,"Delivered CO2 $/t placeholder",2000.0,"t/yr"],
        ["Carbon Dioxide (CO2)","Platform: Conditioning + storage (no capture)","Platform",9,500000,50.0,0.0,0.05,15000,"Wins on uptime and supply assurance; needs offtake",2000.0,"t/yr"],

        ["Ammonia (NH3)","Incumbent: Delivered NH3 + tank","Incumbent",9,300000,0.0,450.0,0.02,5000,"Delivered NH3 $/t placeholder",1000.0,"t/yr"],
        ["Ammonia (NH3)","Platform: NH3 storage + handling + optional cracking","Platform",6,2000000,0.0,0.0,0.05,20000,"Optional resilience buffer; cracking adds capex/energy",1000.0,"t/yr"],
    ]
    return pd.DataFrame(rows, columns=cols)

# -----------------------------
# Streamlit
# -----------------------------
st.set_page_config(page_title="Modular On-Site Gas Utility TEA", layout="wide")
st.title("Unified TEA — Modular On-Site Gas Utility Platform")
st.caption("Compare incumbent delivered gases vs modular on-site platform options (N₂ / O₂ / CO₂ / NH₃).")

assm = default_assumptions()
loads_df = default_loads()
tech_df = default_tech_library()

with st.sidebar:
    st.header("Assumptions")
    assm["electricity_price"] = st.number_input("Electricity price ($/kWh)", value=float(assm["electricity_price"]), step=0.01, format="%.3f")
    assm["wacc"] = st.number_input("WACC", value=float(assm["wacc"]), step=0.01, format="%.3f")
    assm["project_life"] = st.number_input("Project life (years)", value=int(assm["project_life"]), step=1)
    assm["capacity_factor"] = st.number_input("Capacity factor", value=float(assm["capacity_factor"]), step=0.01, format="%.3f")
    assm["contingency_pct"] = st.number_input("Contingency (fraction of installed CAPEX)", value=float(assm["contingency_pct"]), step=0.01, format="%.3f")
    assm["capex_scaling_exponent"] = st.number_input("CAPEX scaling exponent", value=float(assm["capex_scaling_exponent"]), step=0.05, format="%.2f")
    assm["include_shared_labor_in_lcog"] = st.checkbox("Include shared labor inside per-gas LCOG", value=bool(assm["include_shared_labor_in_lcog"]))
    assm["shared_labor_pool"] = st.number_input("Shared labor pool ($/yr)", value=float(assm["shared_labor_pool"]), step=5000.0)

    st.divider()
    st.header("Site scaling")
    scale_mult = st.slider("Scale demand vs anchor site", 0.1, 10.0, 1.0, 0.1)

# Scale loads
loads = loads_df.copy()
for i, r in loads.iterrows():
    if r["unit"] == "Nm3/h" and pd.notnull(r["avg_flow"]):
        loads.loc[i, "avg_flow"] = float(r["avg_flow"]) * scale_mult
    if r["unit"] == "t/yr" and pd.notnull(r["annual_units"]):
        loads.loc[i, "annual_units"] = float(r["annual_units"]) * scale_mult

st.subheader("1) Site loads")
edited_loads = st.data_editor(
    loads[["gas","avg_flow","unit","annual_units","backup_tanks","purity_case"]],
    use_container_width=True,
    num_rows="fixed"
)

st.subheader("2) Technology selection")
gas_list = edited_loads["gas"].tolist()
sel = {}

cols = st.columns(4)
for idx, gas in enumerate(gas_list):
    options = tech_df[tech_df["gas"] == gas]["option"].tolist()
    inc_default = [o for o in options if o.startswith("Incumbent")][0] if any(o.startswith("Incumbent") for o in options) else options[0]
    plat_default = [o for o in options if o.startswith("Platform")][0] if any(o.startswith("Platform") for o in options) else options[-1]
    with cols[idx % 4]:
        st.markdown(f"**{gas}**")
        sel_inc = st.selectbox("Incumbent", options=options, index=options.index(inc_default), key=f"{gas}_inc")
        sel_plat = st.selectbox("Platform", options=options, index=options.index(plat_default), key=f"{gas}_plat")
        sel[gas] = {"incumbent": sel_inc, "platform": sel_plat}

st.subheader("3) Results")

# Determine active gases for shared labor split
active_gases = []
for _, lr in edited_loads.iterrows():
    if lr["unit"] == "Nm3/h":
        v = float(lr["avg_flow"] or 0)
    else:
        v = float(lr["annual_units"] or 0)
    if v > 0:
        active_gases.append(lr["gas"])
n_active = max(1, len(active_gases))
shared_split = 1.0 / n_active

records = []
breakdowns = {}

for _, lr in edited_loads.iterrows():
    gas = lr["gas"]
    load_row = lr.to_dict()
    annual_units = annual_units_from_load(load_row, assm["capacity_factor"])
    demand_value = float(load_row["avg_flow"]) if load_row["unit"] == "Nm3/h" else float(load_row["annual_units"])

    for scenario in ["incumbent", "platform"]:
        opt_name = sel[gas][scenario]
        opt = tech_df[(tech_df["gas"] == gas) & (tech_df["option"] == opt_name)].iloc[0].to_dict()

        base_demand = float(opt.get("base_demand_value", demand_value if demand_value > 0 else 1.0))
        capex_scaled = scale_capex(float(opt["capex_installed"]), demand_value, base_demand, assm["capex_scaling_exponent"])

        lcog_val, br = lcog(
            annual_units=annual_units,
            capex_installed=capex_scaled,
            wacc=assm["wacc"],
            project_life=int(assm["project_life"]),
            contingency_pct=assm["contingency_pct"],
            fixed_om_pct=float(opt["fixed_om_pct"]),
            labor_per_year=float(opt["labor_per_year"]),
            electricity_price=assm["electricity_price"],
            kwh_per_unit=float(opt["kwh_per_unit"]),
            variable_opex_per_unit=float(opt["var_opex_per_unit"]),
            include_shared_labor=assm["include_shared_labor_in_lcog"],
            shared_labor_pool=assm["shared_labor_pool"],
            shared_labor_split=shared_split,
        )

        records.append({
            "Gas": gas,
            "Scenario": scenario.title(),
            "Option": opt_name,
            "TRL": int(opt["trl"]),
            "Demand (avg)": demand_value,
            "Demand unit": load_row["unit"],
            "Annual units": annual_units,
            "CAPEX installed ($)": capex_scaled,
            "LCOG ($/unit)": lcog_val,
            "Avg power (kW)": avg_power_kw(load_row, float(opt["kwh_per_unit"])),
        })
        breakdowns[(gas, scenario)] = br

res_df = pd.DataFrame(records)

def site_total_annual_cost(scenario: str) -> float:
    df = res_df[res_df["Scenario"] == scenario.title()]
    tot = 0.0
    for gas in df["Gas"].unique():
        tot += breakdowns.get((gas, scenario), {}).get("Total annual", 0.0)
    # If shared labor not inside per-gas LCOG, add once at site level
    if not assm["include_shared_labor_in_lcog"] and len(active_gases) > 0:
        tot += assm["shared_labor_pool"]
    return tot

inc_total = site_total_annual_cost("incumbent")
plat_total = site_total_annual_cost("platform")
annual_savings = inc_total - plat_total

inc_capex = res_df[res_df["Scenario"] == "Incumbent"]["CAPEX installed ($)"].sum() * (1 + assm["contingency_pct"])
plat_capex = res_df[res_df["Scenario"] == "Platform"]["CAPEX installed ($)"].sum() * (1 + assm["contingency_pct"])
delta_capex = plat_capex - inc_capex
payback = (delta_capex / annual_savings) if (annual_savings > 0 and delta_capex > 0) else float("nan")
total_power_mw = res_df[res_df["Scenario"] == "Platform"]["Avg power (kW)"].sum() / 1000.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total annual cost (Incumbent)", f"${inc_total:,.0f}/yr")
c2.metric("Total annual cost (Platform)", f"${plat_total:,.0f}/yr")
c3.metric("Annual savings", f"${annual_savings:,.0f}/yr")
c4.metric("Platform electric load", f"{total_power_mw:.3f} MW")

pivot = res_df.pivot_table(index="Gas", columns="Scenario", values="LCOG ($/unit)", aggfunc="first").reset_index()
pivot["Savings (%)"] = (1 - (pivot["Platform"] / pivot["Incumbent"])) * 100
st.markdown("### Per-gas LCOG")
st.dataframe(pivot, use_container_width=True)

st.markdown("### LCOG comparison")
fig, ax = plt.subplots()
x = np.arange(len(pivot["Gas"]))
w = 0.35
ax.bar(x - w/2, pivot["Incumbent"], w, label="Incumbent")
ax.bar(x + w/2, pivot["Platform"], w, label="Platform")
ax.set_xticks(x)
ax.set_xticklabels(pivot["Gas"], rotation=25, ha="right")
ax.set_ylabel("LCOG ($/unit)")
ax.legend()
st.pyplot(fig, use_container_width=True)

st.markdown("### Cost breakdown (pick a gas + scenario)")
b1, b2 = st.columns(2)
gas_pick = b1.selectbox("Gas", options=gas_list, index=0)
scen_pick = b2.selectbox("Scenario", options=["Incumbent", "Platform"], index=1)

br = breakdowns.get((gas_pick, scen_pick.lower()), {})
if br:
    br_df = pd.DataFrame([{"Cost item": k, "Annual $": v} for k, v in br.items()])
    st.dataframe(br_df, use_container_width=True)
else:
    st.info("No breakdown available (likely annual units = 0).")

st.subheader("4) Export")
csv = res_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results.csv", data=csv, file_name="results.csv", mime="text/csv")
