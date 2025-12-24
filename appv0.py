import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Li-Cor 6800 Data Visualizer",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Li-Cor 6800 Data Visualizer")

# =============================================================================
# Session state
# =============================================================================
if "vlines" not in st.session_state:
    st.session_state.vlines = []

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
    for c in df.columns:
        if c.strip().upper() == "TIME":
            return c
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 10 and s.between(1e9, 2e9).mean() > 0.9:
            return c
    return None


# =============================================================================
# Data loader
# =============================================================================
def load_data(file):
    raw = pd.read_excel(file, header=14)
    raw.columns = make_unique(raw.columns)

    units_row = raw.iloc[0]
    units = {c: str(units_row[c]).strip() for c in raw.columns}

    df = raw.iloc[1:].reset_index(drop=True)

    time_col = detect_time_column(df)
    if time_col is None:
        st.error("TIME (Unix seconds) column not found.")
        return None, None

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
    if df is None:
        st.stop()

    if "obs" not in df.columns:
        st.error("Column 'obs' not found (required for vertical lines).")
        st.stop()

    time_col = "TIME"

    # ---- Time slider (snapped)
    times = df[time_col].tolist()
    time_labels = [t.strftime("%H:%M:%S") for t in times]

    st.sidebar.header("Time filter")

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

    # ---- Variable selection (DEFAULT: A & gsw)
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

    # ---- Figure options
    st.sidebar.header("Figure options")
    fig_title = st.sidebar.text_input("Figure title")
    caption = st.sidebar.text_area("Figure caption")

    # ---- Vertical lines
    st.sidebar.subheader("Vertical lines (OBS-based)")

    with st.sidebar.form("add_vline"):
        obs_val = st.number_input("OBS number", step=1)
        label = st.text_input("Label")
        submitted = st.form_submit_button("Add vertical line")

        if submitted:
            match = df[df["obs"] == obs_val]
            if match.empty:
                st.sidebar.error("OBS not found.")
            else:
                st.session_state.vlines.append({
                    "obs": obs_val,
                    "time": match.iloc[0][time_col],
                    "label": label or f"OBS {obs_val}"
                })

    for i, v in enumerate(st.session_state.vlines):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(v["label"])
        if col2.button("❌", key=f"del_{i}"):
            st.session_state.vlines.pop(i)
            st.rerun()

    # =============================================================================
    # Plot
    # =============================================================================


    # =============================================================================
    # Figure font & layout constants (adjust freely)
    # =============================================================================
    TITLE_FONT_SIZE = 30
    AXIS_LABEL_FONT_SIZE = 28
    TICK_FONT_SIZE = 20
    LEGEND_FONT_SIZE = 24
    CAPTION_FONT_SIZE = 20

    PADDING_TOP = 80
    PADDING_BOTTOM = 140
    PADDING_LEFT = 90
    PADDING_RIGHT = 90

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_f[time_col], y=df_f[y_left], name=y_left),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=df_f[time_col], y=df_f[y_right], name=y_right),
        secondary_y=True
    )

    # ---- Vertical lines (robust + non-overlapping labels)
    for i, v in enumerate(st.session_state.vlines):
        xval = v["time"].isoformat()

        fig.add_shape(
            type="line",
            x0=xval,
            x1=xval,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dash", width=1)
        )

        y_pos = 1 - (i % 5) * 0.06  # stagger labels

        fig.add_annotation(
            x=xval,
            y=y_pos,
            xref="x",
            yref="paper",
            text=v["label"],
            showarrow=False,
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)"
        )

    # ---- Caption inside figure (PNG-safe)
    if caption:
        fig.add_annotation(
            text=caption,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.25,
            showarrow=False,
            align="left",
            font=dict(size=CAPTION_FONT_SIZE)
        )

    fig.update_xaxes(
        title_text="Time",
        tickformat="%H:%M:%S",
        title_font=dict(size=AXIS_LABEL_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        showgrid=False
    )

    fig.update_yaxes(
        title_text=f"{y_left} ({units.get(y_left, '')})",
        title_font=dict(size=AXIS_LABEL_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        secondary_y=False,
        showgrid=False
    )

    fig.update_yaxes(
        title_text=f"{y_right} ({units.get(y_right, '')})",
        title_font=dict(size=AXIS_LABEL_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        secondary_y=True,
        showgrid=False
    )

    # 🔹 REMOVE GRID LINES
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.update_layout(
        title=dict(
            text=fig_title,
            x=0.5,                 # center title
            xanchor="center",
            font=dict(size=TITLE_FONT_SIZE)
        ),
        legend=dict(
            font=dict(size=LEGEND_FONT_SIZE)
        ),
        hovermode="x unified",
        height=700,
        margin=dict(
            t=PADDING_TOP,
            b=PADDING_BOTTOM,
            l=PADDING_LEFT,
            r=PADDING_RIGHT
        )
    )

    # ---- Auto PNG filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "toImageButtonOptions": {
            "filename": f"li-cor6800DV_{ts}",
            "scale": 2
        }
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

    with st.expander("Raw data"):
        st.dataframe(df_f)

else:
    st.info("Upload a Li-Cor 6800 Excel file to begin.")
