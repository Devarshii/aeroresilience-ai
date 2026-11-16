
from pathlib import Path
import html
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# AERORESILIENCE AI
# EXECUTIVE DECISION COMMAND CENTER
# COMPLETE APP.PY — SINGLE-LINE HEADING
# ============================================================

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "outputs" / "reports"
DATA = ROOT / "data"

BG = "#07111F"
PANEL = "#101F33"
TEAL = "#27D9C0"
BLUE = "#60A5FA"
WHITE = "#F1F7FF"
AMBER = "#F5B44B"
GRAY = "#71869F"

HERO_IMAGE = (
    "https://images.unsplash.com/"
    "photo-1436491865332-7a61a109cc05"
    "?auto=format&fit=crop&w=1800&q=85"
)

st.set_page_config(
    page_title="AeroResilience AI | Command Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    f"""
<style>
.stApp {{
    background:
        radial-gradient(
            circle at 82% 0%,
            rgba(16,79,104,.23),
            transparent 33%
        ),
        {BG};
    color: {WHITE};
}}

[data-testid="stSidebar"] {{
    background: #0D1A2C;
    border-right: 1px solid #233951;
}}

[data-testid="stHeader"] {{
    background: rgba(7,17,31,.82);
}}

.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1520px;
}}

h1, h2, h3 {{
    color: {WHITE} !important;
    letter-spacing: -.025em;
}}

p, label {{
    color: #B9CCDC;
}}

[data-testid="stMetric"] {{
    background: linear-gradient(
        135deg,
        #15283C,
        #102034
    );
    border: 1px solid #2B4256;
    padding: 18px;
    border-radius: 15px;
    min-height: 115px;
}}

[data-testid="stMetricLabel"] {{
    color: #AFC3D5;
}}

[data-testid="stMetricValue"] {{
    color: #F8FAFC;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid #243951;
    border-radius: 12px;
}}

.aviation-hero {{
    position: relative;
    overflow: hidden;
    min-height: 310px;
    border-radius: 20px;
    border: 1px solid #315367;
    padding: 38px 42px;
    background-image:
        linear-gradient(
            90deg,
            rgba(4,15,30,.97) 0%,
            rgba(5,19,34,.90) 41%,
            rgba(5,21,38,.34) 100%
        ),
        url('{HERO_IMAGE}');
    background-position: center, center 44%;
    background-size: cover;
    box-shadow: 0 22px 50px rgba(0,0,0,.20);
}}

.hero-eyebrow {{
    font-size: 12px;
    color: #5DF3D6;
    font-weight: 750;
    letter-spacing: 2.2px;
}}

.hero-heading {{
    color: white;
    font-size: clamp(28px, 3.4vw, 50px);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.15;
    margin-top: 20px;
    max-width: 100%;
    white-space: nowrap;
}}

.hero-description {{
    color: #CFDFEB;
    max-width: 510px;
    font-size: 15px;
    line-height: 1.65;
    margin-top: 16px;
}}

.hero-tag {{
    display: inline-block;
    margin-top: 20px;
    padding: 8px 13px;
    border: 1px solid #397F86;
    border-radius: 30px;
    color: #A7F3E8;
    background: rgba(17,74,79,.45);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}

.hero-plane {{
    position: absolute;
    right: 7%;
    top: 24%;
    color: rgba(232,252,255,.38);
    font-size: 92px;
    transform: rotate(-23deg);
    text-shadow: 0 0 55px #46E0D5;
}}

.kicker {{
    color: #5DEBD6;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 1.6px;
    margin-top: 15px;
    margin-bottom: 7px;
}}

.insight {{
    border-left: 3px solid {TEAL};
    background: #102338;
    padding: 17px 20px;
    border-radius: 9px;
    margin-top: 12px;
    color: #D2E6F1;
    line-height: 1.7;
}}

.aircraft-card {{
    background: #102238;
    border: 1px solid #29475B;
    border-radius: 13px;
    padding: 16px;
    margin-bottom: 10px;
}}

.aircraft-card-title {{
    color: {TEAL};
    font-weight: 750;
    font-size: 12px;
    letter-spacing: 1.1px;
}}

.aircraft-card-body {{
    color: #C5D7E6;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 8px;
}}

.warning-note {{
    border: 1px solid #765A34;
    border-radius: 9px;
    padding: 13px 16px;
    background: #292219;
    color: #EBD5AF;
    margin: 15px 0;
    font-size: 13px;
}}

@media (max-width:800px) {{
    .aviation-hero {{
        min-height: 250px;
        padding: 26px 22px;
    }}

    .hero-heading {{
        font-size: clamp(20px, 5vw, 34px);
    }}

    .hero-plane {{
        display: none;
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(name):

    path = REPORTS / name

    if not path.is_file():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_col(df, candidates):

    if df.empty:
        return None

    lookup = {
        re.sub(r"[^a-z0-9]", "", str(c).lower()): c
        for c in df.columns
    }

    for candidate in candidates:

        key = re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower(),
        )

        if key in lookup:
            return lookup[key]

    return None


def policy_row(df, policy):

    if df.empty or "Policy" not in df.columns:
        return None

    subset = df[
        df["Policy"].astype(str) == policy
    ]

    return subset.iloc[0] if not subset.empty else None


def num(row, column):

    if row is None or column not in row.index:
        return None

    value = pd.to_numeric(
        row[column],
        errors="coerce",
    )

    return None if pd.isna(value) else float(value)


def fmt(value, decimals=0):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def money(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"${value:,.0f}"


def metric(label, value, suffix="", decimals=0):

    displayed = (
        fmt(value, decimals) + suffix
        if value is not None
        else "N/A"
    )

    st.metric(label, displayed)


def notice(text):

    st.markdown(
        '<div class="insight">'
        + html.escape(text)
        + "</div>",
        unsafe_allow_html=True,
    )


def methodology_note():

    st.markdown(
        """
<div class="warning-note">
⚠️ RESEARCH SIMULATION: Historical BTS flights are real,
but recovery scenarios, passenger demand, seat inventory,
crew assignments and financial assumptions are simulated.
This is not a live airline operations system.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# CHART HELPERS
# ============================================================

def plot_style(fig, height=430):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color="#CEDDEB"),
        height=height,
        margin=dict(
            l=15,
            r=15,
            t=65,
            b=35,
        ),
        legend=dict(
            orientation="h",
            y=1.14,
        ),
    )

    fig.update_xaxes(
        gridcolor="#2B4056",
    )

    fig.update_yaxes(
        gridcolor="#2B4056",
    )

    return fig


def display_chart(fig, height=430):

    st.plotly_chart(
        plot_style(fig, height),
        use_container_width=True,
    )


def table(df, filename=None):

    if df.empty:
        st.info("This report is not available.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    if filename:
        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False),
            file_name=filename,
            mime="text/csv",
            key="download_" + filename,
        )


def bar_compare(labels, values, title, ytitle):

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[
                GRAY if label == "FCFS" else TEAL
                for label in labels
            ],
            text=[
                f"{value:,.1f}"
                for value in values
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=title,
        yaxis_title=ytitle,
        showlegend=False,
    )

    display_chart(fig)


def plot_report(df, key):

    if df.empty:
        st.info("No visualization data available.")
        return

    numeric = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical = [
        col for col in df.columns
        if col not in numeric
    ]

    if not numeric:
        table(df.head(30))
        return

    measure = st.selectbox(
        "Measure",
        numeric,
        key=key + "_measure",
    )

    if categorical:

        category = st.selectbox(
            "Category",
            categorical,
            key=key + "_category",
        )

        grouped = (
            df.groupby(
                category,
                dropna=False,
            )[measure]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )

        grouped[category] = (
            grouped[category].astype(str)
        )

        fig = px.bar(
            grouped,
            x=category,
            y=measure,
            color_discrete_sequence=[TEAL],
            title="Historical Report Explorer",
        )

    else:

        fig = px.line(
            df.reset_index(),
            x="index",
            y=measure,
            color_discrete_sequence=[TEAL],
        )

    display_chart(fig)


# ============================================================
# AIRPORT REFERENCE COORDINATES
# ============================================================

AIRPORTS = {
    "ATL": (33.6407, -84.4277),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "LAX": (33.9416, -118.4085),
    "SFO": (37.6213, -122.3790),
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "BOS": (42.3656, -71.0096),
    "DCA": (38.8512, -77.0402),
    "IAD": (38.9531, -77.4565),
    "MIA": (25.7954, -80.2901),
    "FLL": (26.0726, -80.1527),
    "MCO": (28.4312, -81.3081),
    "TPA": (27.9755, -82.5332),
    "CLT": (35.2140, -80.9431),
    "PHX": (33.4342, -112.0116),
    "LAS": (36.0840, -115.1537),
    "SEA": (47.4502, -122.3088),
    "PDX": (45.5898, -122.5951),
    "SLC": (40.7899, -111.9791),
    "IAH": (29.9902, -95.3368),
    "HOU": (29.6454, -95.2789),
    "MSP": (44.8848, -93.2223),
    "DTW": (42.2162, -83.3554),
    "PHL": (39.8744, -75.2424),
    "BWI": (39.1754, -76.6684),
    "SAN": (32.7338, -117.1933),
    "SJC": (37.3639, -121.9289),
    "SMF": (38.6954, -121.5908),
    "AUS": (30.1975, -97.6664),
    "SAT": (29.5337, -98.4698),
    "STL": (38.7487, -90.3700),
    "BNA": (36.1245, -86.6782),
    "RDU": (35.8776, -78.7875),
    "PIT": (40.4915, -80.2329),
    "CLE": (41.4117, -81.8498),
    "CMH": (39.9980, -82.8919),
    "IND": (39.7173, -86.2944),
    "MCI": (39.2976, -94.7139),
    "MSY": (29.9934, -90.2580),
    "SNA": (33.6757, -117.8682),
    "OAK": (37.7126, -122.2197),
    "HNL": (21.3187, -157.9225),
}


# ============================================================
# HISTORICAL ROUTE PREPARATION
# ============================================================

def prepare_routes(
    df,
    source,
    count_candidates=None,
):

    origin = find_col(
        df,
        [
            "Origin",
            "OriginAirport",
            "OriginCode",
        ],
    )

    dest = find_col(
        df,
        [
            "Dest",
            "Destination",
            "DestAirport",
            "DestinationAirport",
            "DestinationCode",
        ],
    )

    if not origin or not dest:
        return pd.DataFrame()

    result = pd.DataFrame({
        "OriginCode": (
            df[origin]
            .astype(str)
            .str.upper()
            .str.strip()
        ),
        "DestCode": (
            df[dest]
            .astype(str)
            .str.upper()
            .str.strip()
        ),
    })

    weight_col = find_col(
        df,
        count_candidates or [],
    )

    if weight_col:

        result["Weight"] = pd.to_numeric(
            df[weight_col],
            errors="coerce",
        ).fillna(0)

    else:

        result["Weight"] = 1.0

    result = result[
        result["OriginCode"].isin(AIRPORTS)
        & result["DestCode"].isin(AIRPORTS)
        & (
            result["OriginCode"]
            != result["DestCode"]
        )
        & (
            result["Weight"] > 0
        )
    ].copy()

    if result.empty:
        return pd.DataFrame()

    result = (
        result.groupby(
            ["OriginCode", "DestCode"],
            as_index=False,
        )["Weight"]
        .sum()
        .sort_values(
            "Weight",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["Source"] = source

    return result


# ============================================================
# HISTORICAL FLIGHT NETWORK DATA LOADER
# ============================================================

@st.cache_data(show_spinner=False)
def get_network_routes():

    # Priority 1: Existing ORD recovery flight baseline.

    baseline_file = (
        REPORTS / "recovery_baseline_flights.csv"
    )

    if baseline_file.exists():

        try:

            baseline = pd.read_csv(
                baseline_file,
                usecols=["Origin", "Dest"],
            )

            routes = prepare_routes(
                baseline,
                "recovery_baseline_flights.csv",
            )

            if not routes.empty:
                return routes

        except Exception:
            pass

    # Priority 2: Historical route disruption report.

    route_report = load_csv(
        "route_disruptions.csv"
    )

    routes = prepare_routes(
        route_report,
        "route_disruptions.csv",
        [
            "Flights",
            "FlightCount",
            "TotalFlights",
            "NumberOfFlights",
        ],
    )

    if not routes.empty:
        return routes

    # Priority 3: Other existing recovery schedules.

    recovery_candidates = [
        "optimized_recovery_flights.csv",
        "step32_recovery_schedules.csv",
        "aircraft_aware_optimized_schedule.csv",
    ]

    for filename in recovery_candidates:

        path = REPORTS / filename

        if not path.exists():
            continue

        try:

            header = pd.read_csv(
                path,
                nrows=0,
            )

            origin = find_col(
                header,
                ["Origin"],
            )

            dest = find_col(
                header,
                ["Dest"],
            )

            if not origin or not dest:
                continue

            data = pd.read_csv(
                path,
                usecols=[origin, dest],
            )

            routes = prepare_routes(
                data,
                filename,
            )

            if not routes.empty:
                return routes

        except Exception:
            continue

    # Priority 4: Existing flight CSV/Parquet datasets.

    candidates = []

    for folder in [
        DATA / "processed",
        DATA / "raw",
    ]:

        if not folder.exists():
            continue

        candidates.extend(
            folder.glob("*.parquet")
        )

        candidates.extend(
            folder.glob("*.csv")
        )

    candidates = [
        path
        for path in candidates
        if not any(
            term in path.name.lower()
            for term in [
                "passenger",
                "crew",
                "taxi",
                "yellow_tripdata",
            ]
        )
    ]

    for path in sorted(candidates):

        try:

            if path.suffix == ".parquet":

                import pyarrow.parquet as pq

                schema = pq.read_schema(path)

                header = pd.DataFrame(
                    columns=schema.names
                )

                origin = find_col(
                    header,
                    ["Origin"],
                )

                dest = find_col(
                    header,
                    ["Dest"],
                )

                if not origin or not dest:
                    continue

                data = pd.read_parquet(
                    path,
                    columns=[origin, dest],
                )

            elif path.suffix == ".csv":

                header = pd.read_csv(
                    path,
                    nrows=0,
                )

                origin = find_col(
                    header,
                    ["Origin"],
                )

                dest = find_col(
                    header,
                    ["Dest"],
                )

                if not origin or not dest:
                    continue

                data = pd.read_csv(
                    path,
                    usecols=[origin, dest],
                )

            else:
                continue

            routes = prepare_routes(
                data,
                path.name,
            )

            if not routes.empty:
                return routes

        except Exception:
            continue

    return pd.DataFrame()


# ============================================================
# INTERACTIVE U.S. FLIGHT NETWORK MAP
# ============================================================

def flight_network_map():

    routes = get_network_routes()

    if routes.empty:

        st.info(
            "No compatible historical airport routes "
            "were found. No connections have been invented."
        )

        return

    max_routes = min(
        150,
        len(routes),
    )

    count = st.slider(
        "Historical routes to display",
        min_value=1,
        max_value=max_routes,
        value=min(45, max_routes),
        key="route_map_slider",
    )

    selected = routes.head(count)

    fig = go.Figure()

    # Route connections.

    for row in selected.itertuples(index=False):

        lat1, lon1 = AIRPORTS[
            row.OriginCode
        ]

        lat2, lon2 = AIRPORTS[
            row.DestCode
        ]

        fig.add_trace(
            go.Scattergeo(
                lon=[lon1, lon2],
                lat=[lat1, lat2],
                mode="lines",
                line=dict(
                    color="rgba(39,217,192,.52)",
                    width=1.5,
                ),
                text=[
                    (
                        f"{row.OriginCode} → "
                        f"{row.DestCode}"
                    )
                ] * 2,
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Airport markers.

    codes = sorted(
        set(selected["OriginCode"])
        | set(selected["DestCode"])
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[
                AIRPORTS[code][0]
                for code in codes
            ],
            lon=[
                AIRPORTS[code][1]
                for code in codes
            ],
            text=codes,
            mode="markers+text",
            textposition="top center",
            textfont=dict(
                size=10,
                color="#E5F8FF",
            ),
            marker=dict(
                size=7,
                color=TEAL,
                line=dict(
                    width=1,
                    color="#A5FFF0",
                ),
            ),
            hovertemplate=(
                "Airport: %{text}<extra></extra>"
            ),
            name="Airports",
        )
    )

    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#152D42",
        showlakes=True,
        lakecolor="#0A182B",
        showocean=True,
        oceancolor="#0A182B",
        showsubunits=True,
        subunitcolor="#426078",
        showcoastlines=True,
        coastlinecolor="#426078",
        bgcolor=BG,
    )

    fig.update_layout(
        title="U.S. Airport & Historical Route Network",
        paper_bgcolor=BG,
        font=dict(color=WHITE),
        height=520,
        margin=dict(
            l=0,
            r=0,
            t=65,
            b=0,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        f"Source: {routes.iloc[0]['Source']} | "
        f"{len(selected)} route pairs displayed. "
        "Routes are historical or scenario-input "
        "flight connections, not live positions."
    )


# ============================================================
# LOAD ANALYTICAL REPORTS
# ============================================================

finance = load_csv(
    "recovery_financial_cost_baseline.csv"
)

passengers = load_csv(
    "passenger_rebooking_summary.csv"
)

aircraft = load_csv(
    "integrated_resilience_comparison.csv"
)

crew = load_csv(
    "crew_resilience_comparison.csv"
)

connections = load_csv(
    "passenger_connection_resilience_comparison.csv"
)

what_if = load_csv(
    "resilience_what_if_scenarios.csv"
)

financial_sensitivity = load_csv(
    "financial_sensitivity_scenarios.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ✈️ AERORESILIENCE"
    )

    st.caption(
        "AI AIRLINE OPERATIONS COMMAND CENTER"
    )

    st.divider()

    pages = [
        "Executive Overview",
        "Flight Disruption Intelligence",
        "Aircraft Recovery",
        "Crew Recovery",
        "Passenger Intelligence",
        "Financial Analytics",
        "What-If Simulator",
    ]

    page = st.radio(
        "NAVIGATION",
        pages,
        index=0,
    )

    st.divider()

    st.markdown(
        "**SYSTEM STATUS**"
    )

    st.success(
        "5 analytical prototypes complete"
    )

    st.info(
        "Executive dashboard — in development"
    )

    st.caption(
        "DATA: Historical BTS flight schedules"
    )

    st.caption(
        "MODE: Research simulation"
    )

    st.caption(
        "Not connected to real-time airline systems."
    )


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    # FIX: Heading displays on ONE line.
    # HTML is unindented to prevent code-block rendering.

    st.markdown(
        """
<div class="aviation-hero">
<div class="hero-eyebrow">AIRLINE OPERATIONS INTELLIGENCE</div>
<div class="hero-heading">AeroResilience AI</div>
<div class="hero-description">
Predict disruption. Optimize aircraft and crew recovery.
Protect passenger journeys. Evaluate the financial
impact of every modeled recovery decision.
</div>
<div class="hero-tag">✦ EXECUTIVE DECISION COMMAND CENTER</div>
<div class="hero-plane">✈</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="kicker">OPERATIONAL INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    fcfs = policy_row(
        finance,
        "FCFS",
    )

    aware = policy_row(
        finance,
        "AircraftAware",
    )

    fdelay = num(
        fcfs,
        "IncrementalNetworkDelayMinutes",
    )

    adelay = num(
        aware,
        "IncrementalNetworkDelayMinutes",
    )

    fcost = num(
        fcfs,
        "TotalModeledRecoveryCostUSD",
    )

    acost = num(
        aware,
        "TotalModeledRecoveryCostUSD",
    )

    improvement = (
        100 * (fdelay - adelay) / fdelay
        if (
            fdelay is not None
            and adelay is not None
            and fdelay > 0
        )
        else None
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "✈️ Historical Flights",
            "1,847,936",
        )

    with c2:
        metric(
            "🛬 Network Delay Reduction",
            improvement,
            "%",
            2,
        )

    with c3:
        metric(
            "💺 FCFS Rebooking Rate",
            num(
                policy_row(
                    passengers,
                    "FCFS",
                ),
                "RebookingRatePercent",
            ),
            "%",
            2,
        )

    with c4:
        st.metric(
            "💰 Illustrative Cost Difference",
            money(
                fcost - acost
                if (
                    fcost is not None
                    and acost is not None
                )
                else None
            ),
        )

    st.markdown(
        "### 🗺️ U.S. Flight Network"
    )

    flight_network_map()

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### ✈️ Aircraft Recovery"
        )

        if (
            fdelay is not None
            and adelay is not None
        ):

            bar_compare(
                ["FCFS", "AircraftAware"],
                [fdelay, adelay],
                "Simulated Incremental Network Delay",
                "Minutes",
            )

        else:

            st.info(
                "Aircraft recovery baseline unavailable."
            )

    with right:

        st.markdown(
            "### 💺 Passenger Recovery"
        )

        pf = policy_row(
            passengers,
            "FCFS",
        )

        pa = policy_row(
            passengers,
            "AircraftAware",
        )

        pf_count = num(
            pf,
            "PassengersRebooked",
        )

        pa_count = num(
            pa,
            "PassengersRebooked",
        )

        if (
            pf_count is not None
            and pa_count is not None
        ):

            bar_compare(
                ["FCFS", "AircraftAware"],
                [pf_count, pa_count],
                "Synthetic Passengers Rebooked",
                "Passengers",
            )

        else:

            st.info(
                "Passenger rebooking results unavailable."
            )

    notice(
        "KEY EXECUTIVE INSIGHT — Aircraft-aware recovery "
        "reduced modeled network delay, but did not "
        "produce better passenger rebooking outcomes "
        "in the tested scenario. Recovery decisions "
        "must balance multiple objectives."
    )

    methodology_note()


# ============================================================
# PAGE 2 — FLIGHT DISRUPTION INTELLIGENCE
# ============================================================

elif page == "Flight Disruption Intelligence":

    st.title(
        "✈️ Flight Disruption Intelligence"
    )

    st.caption(
        "Module 1 — Historical flight risk, "
        "routes and alert prioritization"
    )

    st.markdown(
        '<div class="kicker">HISTORICAL NETWORK ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    flight_network_map()

    tabs = st.tabs([
        "Route Disruptions",
        "Flight Alerts",
        "Airport Analysis",
    ])

    with tabs[0]:

        routes = load_csv(
            "route_disruptions.csv"
        )

        if not routes.empty:

            plot_report(
                routes,
                "route",
            )

            table(
                routes,
                "route_disruptions.csv",
            )

        else:

            st.info(
                "Route report unavailable."
            )

    with tabs[1]:

        alerts = load_csv(
            "topk_alert_metrics.csv"
        )

        table(
            alerts,
            "topk_alert_metrics.csv",
        )

    with tabs[2]:

        options = [
            "validated_cascade_airports.csv",
            "stress_test_airport_summary.csv",
            "threshold_comparison.csv",
        ]

        selected_report = st.selectbox(
            "Choose an existing report",
            options,
        )

        data = load_csv(
            selected_report
        )

        if not data.empty:

            plot_report(
                data,
                "airport_report",
            )

        table(
            data,
            selected_report,
        )

    st.caption(
        "Historical reports are not live flight alerts."
    )


# ============================================================
# PAGE 3 — AIRCRAFT RECOVERY
# ============================================================

elif page == "Aircraft Recovery":

    st.title(
        "🛫 Aircraft Recovery Optimization"
    )

    st.caption(
        "Module 2 — Cascading delays, aircraft "
        "turnaround and recovery prioritization"
    )

    st.markdown(
        """
<div class="aircraft-card">
<div class="aircraft-card-title">AIRCRAFT ROTATION LOGIC</div>
<div class="aircraft-card-body">
✈ ORD → DFW → DEN → SFO
<br>
A delayed departure may affect later flights
operated by the same aircraft.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Illustrative flight sequence, not an "
        "actual aircraft tracking record."
    )

    if not aircraft.empty:

        turns = sorted(
            aircraft[
                "MinimumTurnMinutes"
            ].dropna().unique().tolist()
        )

        turn = st.select_slider(
            "Minimum aircraft turnaround",
            options=turns,
            format_func=lambda x: f"{x:g} minutes",
        )

        chosen = aircraft[
            aircraft["MinimumTurnMinutes"] == turn
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:

            metric(
                "Delay Minutes Avoided",
                chosen["DelayMinutesAvoided"],
            )

        with c2:

            metric(
                "Network Delay Improvement",
                chosen[
                    "AircraftDelayImprovementPercent"
                ],
                "%",
                2,
            )

        with c3:

            st.metric(
                "Illustrative Cost Difference",
                money(
                    chosen[
                        "IllustrativeCostReductionUSD"
                    ]
                ),
            )

        fig = go.Figure()

        for column, label, color in [
            (
                "FCFSIncrementalDelayMinutes",
                "FCFS",
                GRAY,
            ),
            (
                "AircraftAwareIncrementalDelayMinutes",
                "Aircraft-aware",
                TEAL,
            ),
        ]:

            fig.add_trace(
                go.Scatter(
                    x=aircraft[
                        "MinimumTurnMinutes"
                    ],
                    y=aircraft[column],
                    mode="lines+markers",
                    name=label,
                    line=dict(
                        color=color,
                        width=3,
                    ),
                )
            )

        fig.update_layout(
            title="Aircraft Turnaround Sensitivity",
            xaxis_title=(
                "Minimum turnaround (minutes)"
            ),
            yaxis_title=(
                "Incremental network delay (minutes)"
            ),
        )

        display_chart(fig)

        table(
            aircraft,
            "integrated_resilience_comparison.csv",
        )

    else:

        st.info(
            "Aircraft resilience report unavailable."
        )

    methodology_note()


# ============================================================
# PAGE 4 — CREW RECOVERY
# ============================================================

elif page == "Crew Recovery":

    st.title(
        "👨‍✈️ Crew Recovery & Resilience"
    )

    st.caption(
        "Module 2 + Module 5 — Synthetic crew "
        "availability and stress testing"
    )

    if not crew.empty:

        scenarios = crew[
            "Scenario"
        ].astype(str).tolist()

        scenario = st.selectbox(
            "Crew disruption scenario",
            scenarios,
        )

        row = crew[
            crew["Scenario"].astype(str) == scenario
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:

            metric(
                "FCFS Crew Coverage",
                row["FCFSCoveragePercent"],
                "%",
                2,
            )

        with c2:

            metric(
                "Aircraft-Aware Crew Coverage",
                row[
                    "AircraftAwareCoveragePercent"
                ],
                "%",
                2,
            )

        with c3:

            metric(
                "Aircraft-Aware Uncovered Flights",
                row[
                    "AircraftAwareUncoveredFlights"
                ],
            )

        bar_compare(
            ["FCFS", "AircraftAware"],
            [
                row["FCFSCoveragePercent"],
                row["AircraftAwareCoveragePercent"],
            ],
            "Synthetic Crew Coverage",
            "Coverage (%)",
        )

        st.markdown(
            "### Crew Stress-Test Results"
        )

        table(
            crew,
            "crew_resilience_comparison.csv",
        )

        notice(
            "CREW RESILIENCE — Late arriving reserve "
            "crews can reduce the feasibility of "
            "otherwise effective recovery strategies."
        )

    else:

        st.info(
            "Crew resilience results unavailable."
        )

    methodology_note()


# ============================================================
# PAGE 5 — PASSENGER INTELLIGENCE
# ============================================================

elif page == "Passenger Intelligence":

    st.title(
        "💺 Passenger Recovery Intelligence"
    )

    st.caption(
        "Module 3 — Missed connections, alternative "
        "flights and passenger resilience"
    )

    tabs = st.tabs([
        "Rebooking Outcomes",
        "Connection Sensitivity",
        "Global Optimization",
    ])

    with tabs[0]:

        if not passengers.empty:

            fig = go.Figure()

            for column, label, color in [
                (
                    "PassengersRebooked",
                    "Rebooked",
                    TEAL,
                ),
                (
                    "PassengersUnaccommodated",
                    "Unaccommodated",
                    AMBER,
                ),
            ]:

                fig.add_trace(
                    go.Bar(
                        x=passengers["Policy"],
                        y=passengers[column],
                        name=label,
                        marker_color=color,
                    )
                )

            fig.update_layout(
                title="Passenger Rebooking Comparison",
                barmode="stack",
                yaxis_title="Synthetic passengers",
            )

            display_chart(fig)

            table(
                passengers,
                "passenger_rebooking_summary.csv",
            )

        else:

            st.info(
                "Passenger rebooking report unavailable."
            )

    with tabs[1]:

        if not connections.empty:

            fig = go.Figure()

            for column, label, color in [
                (
                    "FCFSPassengersAtRisk",
                    "FCFS",
                    GRAY,
                ),
                (
                    "AircraftAwarePassengersAtRisk",
                    "Aircraft-aware",
                    TEAL,
                ),
            ]:

                fig.add_trace(
                    go.Scatter(
                        x=connections[
                            "MinimumConnectionMinutes"
                        ],
                        y=connections[column],
                        mode="lines+markers",
                        name=label,
                        line=dict(
                            color=color,
                            width=3,
                        ),
                    )
                )

            fig.update_layout(
                title="Minimum Connection Time Sensitivity",
                xaxis_title=(
                    "Required connection time (minutes)"
                ),
                yaxis_title="Passengers at risk",
            )

            display_chart(fig)

            table(
                connections,
                "passenger_connection_resilience_comparison.csv",
            )

        else:

            st.info(
                "Connection sensitivity report unavailable."
            )

    with tabs[2]:

        global_data = load_csv(
            "global_passenger_optimization_summary.csv"
        )

        table(
            global_data,
            "global_passenger_optimization_summary.csv",
        )

        notice(
            "The global passenger allocation optimizer "
            "maximized rebooking coverage, then minimized "
            "arrival delay. It did not improve on greedy "
            "passenger coverage in our tested case."
        )

    methodology_note()


# ============================================================
# PAGE 6 — FINANCIAL ANALYTICS
# ============================================================

elif page == "Financial Analytics":

    st.title(
        "💰 Financial Impact & Recovery"
    )

    st.caption(
        "Module 4 — Hypothetical costs and "
        "financial sensitivity"
    )

    f = policy_row(
        finance,
        "FCFS",
    )

    a = policy_row(
        finance,
        "AircraftAware",
    )

    fc = num(
        f,
        "TotalModeledRecoveryCostUSD",
    )

    ac = num(
        a,
        "TotalModeledRecoveryCostUSD",
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "FCFS Modeled Cost",
            money(fc),
        )

    with c2:

        st.metric(
            "Aircraft-Aware Modeled Cost",
            money(ac),
        )

    with c3:

        st.metric(
            "Modeled Cost Difference",
            money(
                fc - ac

                if fc is not None
                and ac is not None

                else None
            ),
        )

    if not finance.empty:

        components = {
            "OperationalDelayCostUSD": "Delay",
            "PassengerRebookingCostUSD": "Rebooking",
            "PassengerAssistanceCostUSD": "Assistance",
            "ReserveCrewAssignmentCostUSD": "Crew",
        }

        fig = go.Figure()

        for column, label in components.items():

            if column in finance.columns:

                fig.add_trace(
                    go.Bar(
                        x=finance["Policy"],
                        y=finance[column],
                        name=label,
                    )
                )

        fig.update_layout(
            title="Illustrative Recovery Cost Drivers",
            barmode="stack",
            yaxis_title="USD",
        )

        display_chart(fig)

        table(
            finance,
            "recovery_financial_cost_baseline.csv",
        )

    st.markdown(
        "### Financial Sensitivity"
    )

    if not financial_sensitivity.empty:

        counts = financial_sensitivity[
            "LowerCostPolicy"
        ].value_counts()

        bar_compare(
            counts.index.tolist(),
            counts.values.tolist(),
            "Lower-Cost Policy Across Tested Scenarios",
            "Scenario count",
        )

        table(
            financial_sensitivity,
            "financial_sensitivity_scenarios.csv",
        )

    else:

        st.info(
            "Financial sensitivity report unavailable."
        )

    methodology_note()


# ============================================================
# PAGE 7 — WHAT-IF SIMULATOR
# ============================================================

elif page == "What-If Simulator":

    st.title(
        "🎛️ Resilience & What-If Simulator"
    )

    st.caption(
        "Module 5 — Aircraft turnaround and "
        "hypothetical financial scenarios"
    )

    if not what_if.empty:

        turns = sorted(
            what_if[
                "MinimumTurnMinutes"
            ].dropna().unique().tolist()
        )

        delays = sorted(
            what_if[
                "DelayCostPerMinuteUSD"
            ].dropna().unique().tolist()
        )

        rebooking_rates = sorted(
            what_if[
                "RebookingCostPerPassengerUSD"
            ].dropna().unique().tolist()
        )

        assistance_rates = sorted(
            what_if[
                "AssistanceCostPerPassengerUSD"
            ].dropna().unique().tolist()
        )

        c1, c2 = st.columns(2)

        with c1:

            turn = st.selectbox(
                "Minimum aircraft turnaround",
                turns,
                format_func=lambda x: f"{x:g} min",
            )

            delay = st.selectbox(
                "Operational delay rate",
                delays,
                index=(
                    delays.index(45)
                    if 45 in delays
                    else 0
                ),
                format_func=lambda x: f"${x:g}/minute",
            )

        with c2:

            rebook = st.selectbox(
                "Passenger rebooking cost",
                rebooking_rates,
                index=(
                    rebooking_rates.index(80)
                    if 80 in rebooking_rates
                    else 0
                ),
                format_func=lambda x: f"${x:g}/passenger",
            )

            assist = st.selectbox(
                "Unaccommodated passenger assistance",
                assistance_rates,
                index=(
                    assistance_rates.index(250)
                    if 250 in assistance_rates
                    else 0
                ),
                format_func=lambda x: f"${x:g}/passenger",
            )

        selected = what_if[
            (
                what_if["MinimumTurnMinutes"] == turn
            )
            & (
                what_if["DelayCostPerMinuteUSD"]
                == delay
            )
            & (
                what_if[
                    "RebookingCostPerPassengerUSD"
                ] == rebook
            )
            & (
                what_if[
                    "AssistanceCostPerPassengerUSD"
                ] == assist
            )
        ]

        if not selected.empty:

            row = selected.iloc[0]

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "FCFS Modeled Cost",
                    money(
                        row["FCFSModeledCostUSD"]
                    ),
                )

            with c2:

                st.metric(
                    "Aircraft-Aware Modeled Cost",
                    money(
                        row[
                            "AircraftAwareModeledCostUSD"
                        ]
                    ),
                )

            with c3:

                st.metric(
                    "Cost Difference",
                    money(
                        row["CostReductionUSD"]
                    ),
                )

            notice(
                "LOWER MODELED COST POLICY — "
                + str(row["LowerCostPolicy"])
                + ". This result applies only to the "
                "selected hypothetical scenario."
            )

            bar_compare(
                ["FCFS", "AircraftAware"],
                [
                    row["FCFSModeledCostUSD"],
                    row["AircraftAwareModeledCostUSD"],
                ],
                "Selected Recovery Cost Scenario",
                "USD",
            )

        st.markdown(
            "### Financial Resilience Heatmap"
        )

        filtered = what_if[
            (
                what_if[
                    "RebookingCostPerPassengerUSD"
                ] == rebook
            )
            & (
                what_if[
                    "AssistanceCostPerPassengerUSD"
                ] == assist
            )
        ]

        matrix = filtered.pivot_table(
            index="MinimumTurnMinutes",
            columns="DelayCostPerMinuteUSD",
            values="CostReductionUSD",
            aggfunc="first",
        )

        if not matrix.empty:

            fig = go.Figure(
                go.Heatmap(
                    x=matrix.columns,
                    y=matrix.index,
                    z=matrix.values,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(
                        title="USD difference"
                    ),
                )
            )

            fig.update_layout(
                title="Scenario Financial Sensitivity",
                xaxis_title="Delay rate ($/minute)",
                yaxis_title=(
                    "Minimum turnaround (minutes)"
                ),
            )

            display_chart(
                fig,
                520,
            )

        with st.expander(
            "View all scenarios"
        ):

            table(
                what_if,
                "resilience_what_if_scenarios.csv",
            )

    else:

        st.info(
            "What-if scenario report unavailable."
        )

    methodology_note()

    st.caption(
        "Scenario controls select precomputed results. "
        "Passenger and crew baseline quantities remain "
        "fixed in financial what-if comparisons."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "✈ AeroResilience AI | Executive Decision Command Center"
)

st.caption(
    "Historical BTS data • Simulated recovery scenarios "
    "• Illustrative financial assumptions • Research prototype"
)
