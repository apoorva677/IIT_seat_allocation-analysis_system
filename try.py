import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# =========================
# LOAD DATA (SAFE)
# =========================
try:
    df = pd.read_csv("JEE_Rank_2016_2024.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please ensure it is uploaded to the GitHub repo.")
    st.stop()

# =========================
# CLEAN DATA (SAFE VERSION)
# =========================
df['Opening_Rank'] = pd.to_numeric(df['Opening_Rank'], errors='coerce')
df['Closing_Rank'] = pd.to_numeric(df['Closing_Rank'], errors='coerce')

# Only drop critical missing fields (NOT ranks)
df = df.dropna(subset=['Year', 'Institute', 'Seat_Type'])

# =========================
# TITLE
# =========================
st.title("IIT Seat Allocation Analysis System")
st.write("Explore trends in seat allocation, gender distribution, and closing ranks across IITs.")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filter Options")

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df['Year'].min()),
    int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

quota_options = st.sidebar.multiselect(
    "Select Quota",
    df['Quota'].dropna().unique(),
    default=df['Quota'].dropna().unique()
)

gender_options = st.sidebar.multiselect(
    "Select Gender",
    df['Gender'].dropna().unique(),
    default=df['Gender'].dropna().unique()
)

# =========================
# FILTER DATA
# =========================
filtered_df = df[
    (df['Year'].between(year_range[0], year_range[1])) &
    (df['Quota'].isin(quota_options)) &
    (df['Gender'].isin(gender_options))
]

st.write("Filtered rows:", len(filtered_df))  # DEBUG HELPER

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "Overall Analysis",
    "Program-Specific Analysis",
    "Multiple Institute Comparison"
])

# =========================
# TAB 1 - OVERALL ANALYSIS
# =========================
with tab1:
    st.header("Overall Analysis")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    else:

        # Programs per institute (RED)
        programs = filtered_df.groupby('Institute')['Academic_Program_Name'].nunique()
        fig = px.bar(
            programs,
            title="Number of Programs per Institute",
            color_discrete_sequence=["red"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Seat distribution (RED)
        seats = filtered_df.groupby('Institute')['Seat_Type'].count()
        fig = px.bar(
            seats,
            title="Seat Distribution Across Institutes",
            color_discrete_sequence=["red"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Gender distribution (RED)
        gender = filtered_df['Gender'].value_counts()
        fig = px.pie(
            values=gender.values,
            names=gender.index,
            title="Gender Distribution",
            color_discrete_sequence=["red", "darkred"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trend: seat intake (RED line)
        seat_year = filtered_df.groupby('Year')['Seat_Type'].count().reset_index()

        model = LinearRegression()
        X = seat_year[['Year']]
        y = seat_year['Seat_Type']
        model.fit(X, y)
        pred = model.predict(X)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seat_year['Year'],
            y=y,
            mode='markers',
            name='Actual',
            marker=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=seat_year['Year'],
            y=pred,
            mode='lines',
            name='Trend',
            line=dict(color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Closing rank trend (RED)
        rank_year = filtered_df.groupby('Year')['Closing_Rank'].mean().reset_index()

        fig = px.line(
            rank_year,
            x='Year',
            y='Closing_Rank',
            title="Average Closing Rank Trend",
            markers=True
        )
        fig.update_traces(line=dict(color="red"))
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data", csv, "filtered_data.csv")

# =========================
# TAB 2 - PROGRAM ANALYSIS
# =========================
with tab2:
    st.header("Program-Specific Analysis")

    if not filtered_df.empty:

        program = st.selectbox(
            "Select Program",
            filtered_df['Academic_Program_Name'].dropna().unique()
        )

        institute = st.selectbox(
            "Select Institute",
            filtered_df['Institute'].dropna().unique()
        )

        data = filtered_df[
            (filtered_df['Academic_Program_Name'] == program) &
            (filtered_df['Institute'] == institute)
        ]

        if data.empty:
            st.warning("No data for selection.")
        else:

            open_rank = data.groupby('Year')['Opening_Rank'].mean().reset_index()
            close_rank = data.groupby('Year')['Closing_Rank'].mean().reset_index()

            fig = px.line(open_rank, x='Year', y='Opening_Rank',
                          title="Opening Rank Trend",
                          markers=True)
            fig.update_traces(line=dict(color="red"))
            st.plotly_chart(fig, use_container_width=True)

            fig = px.line(close_rank, x='Year', y='Closing_Rank',
                          title="Closing Rank Trend",
                          markers=True)
            fig.update_traces(line=dict(color="red"))
            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3 - COMPARISON
# =========================
with tab3:
    st.header("Multiple Institute Comparison")

    top_n = st.slider("Top N", 1, 20, 5)
    type_ = st.selectbox("Compare by", ["Institute", "Program"])

    if type_ == "Institute":
        comp = filtered_df.groupby('Institute').agg(
            avg_open=('Opening_Rank', 'mean'),
            avg_close=('Closing_Rank', 'mean')
        ).reset_index()

    else:
        comp = filtered_df.groupby('Academic_Program_Name').agg(
            avg_open=('Opening_Rank', 'mean'),
            avg_close=('Closing_Rank', 'mean')
        ).reset_index()

    if not comp.empty:
        st.dataframe(comp.head(top_n))

        fig = px.bar(
            comp.head(top_n),
            x=comp.columns[0],
            y='avg_open',
            title="Top Comparison",
            color_discrete_sequence=["red"]
        )
        st.plotly_chart(fig, use_container_width=True)
