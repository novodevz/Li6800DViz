import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Li-Cor 6800 Data Visualizer",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Li-Cor 6800 Photosynthesis Data Analyzer")

# =============================================================================
# Helpers
# =============================================================================
def make_unique(cols):
    out = []
    seen = {}
    for c in cols:
        c = str(c).strip()
        if c == "" or c.lower() == "nan":
            c = "unnamed"
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


def detect_time_column(df):
    # Prefer explicit TIME column
    for c in df.columns:
        if c.strip().upper() == "TIME":
            return c

    # Fallback: detect Unix seconds
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 10:
            continue
        if s.between(1e9, 2e9).mean() > 0.9:
            return c
    return None


# =============================================================================
# Data loader (Li-Cor correct)
# =============================================================================
def load_data(file):
    raw = pd.read_excel(file, header=14)
    raw.columns = make_unique(raw.columns)

    # Units row (row immediately after header)
    units_row = raw.iloc[0]
    units = {c: str(units_row[c]).strip() for c in raw.columns}

    # Actual data starts immediately after units
    df = raw.iloc[1:].reset_index(drop=True)

    time_col = detect_time_column(df)
    if time_col is None:
        st.error("TIME (Unix seconds) column not found.")
        return None, None

    # Convert TIME
    t_num = pd.to_numeric(df[time_col], errors="coerce")
    t_dt = pd.to_datetime(t_num, unit="s", utc=True, errors="coerce")
    t_dt = t_dt.dt.tz_convert("Asia/Jerusalem")

    for c in df.columns:
        if c != time_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["TIME"] = t_dt
    df = df.dropna(subset=["TIME"]).sort_values("TIME")

    df.columns = make_unique(df.columns)

    return df, units


# =============================================================================
# App
# =============================================================================
uploaded = st.sidebar.file_uploader("Upload Li-Cor 6800 .xlsx", type=["xlsx"])

if uploaded:
    df, units = load_data(uploaded)

    if df is not None:
        time_col = "TIME"

        # ---- Snap slider to ACTUAL timestamps
        times = df[time_col].tolist()
        time_labels = [t.strftime("%H:%M:%S") for t in times]

        st.sidebar.header("Time filter (Jerusalem, snapped)")

        idx_start, idx_end = st.sidebar.select_slider(
            "Select time range",
            options=list(range(len(times))),
            value=(0, len(times) - 1),
            format_func=lambda i: time_labels[i]
        )

        df_f = df.iloc[idx_start: idx_end + 1]

        if df_f.empty:
            st.warning("No data in selected range.")
            st.stop()

        # ---- Variable selection (headers only)
        selectable = [c for c in df.columns if c != time_col]

        def default_pick(names):
            for n in names:
                for c in selectable:
                    if c.lower() == n.lower():
                        return c
            return selectable[0]

        left_def = default_pick(["A", "Photo"])
        right_def = default_pick(["gsw", "Cond"])

        c1, c2 = st.columns(2)
        with c1:
            y_left = st.selectbox("Left Y-axis", selectable, index=selectable.index(left_def))
        with c2:
            y_right = st.selectbox("Right Y-axis", selectable, index=selectable.index(right_def))

        # ---- Plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df_f[time_col],
                y=df_f[y_left],
                name=y_left
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df_f[time_col],
                y=df_f[y_right],
                name=y_right
            ),
            secondary_y=True
        )

        fig.update_xaxes(
            title_text="Time (Jerusalem)",
            tickformat="%H:%M:%S"
        )

        fig.update_yaxes(
            title_text=f"{y_left} ({units.get(y_left, '')})",
            secondary_y=False
        )

        fig.update_yaxes(
            title_text=f"{y_right} ({units.get(y_right, '')})",
            secondary_y=True
        )

        fig.update_layout(
            hovermode="x unified",
            height=650
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw data"):
            st.dataframe(df_f)

else:
    st.info("Upload a Li-Cor 6800 Excel file to begin.")
