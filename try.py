import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
st.write(os.listdir())

# Load the dataset
df = pd.read_csv("JEE_Rank_2016_2024.csv")
df.dropna(inplace=True)
df['Opening_Rank'] = pd.to_numeric(df['Opening_Rank'], errors='coerce')
df['Closing_Rank'] = pd.to_numeric(df['Closing_Rank'], errors='coerce')

# Title and description
st.title("IIT Seat Allocation Analysis System")
st.write("Explore trends in seat allocation, gender distribution, and closing ranks across IITs.")

# Sidebar filters (common across tabs)
st.sidebar.header("Filter Options")
year_range = st.sidebar.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (2016, 2024))
quota_options = st.sidebar.multiselect("Select Quota", df['Quota'].unique(), default=df['Quota'].unique())
gender_options = st.sidebar.multiselect("Select Gender", df['Gender'].unique(), default=df['Gender'].unique())

# Filter data based on selections
filtered_df = df[(df['Year'].between(year_range[0], year_range[1])) & 
                 (df['Quota'].isin(quota_options)) & 
                 (df['Gender'].isin(gender_options))]

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["Overall Analysis", "Program-Specific Analysis", "Multiple Institute Comparison"])

# Tab 1: Overall Analysis
with tab1:
    st.header("Overall Analysis")
    
    # Graph 1: Number of programs per institute (Interactive with Plotly)
    programs_per_institute = filtered_df.groupby('Institute')['Academic_Program_Name'].nunique()
    st.write("### Number of Programs per Institute")
    fig = px.bar(programs_per_institute.sort_values(), 
                 labels={'index': 'Institute', 'value': 'Number of Programs'}, 
                 title="Number of Programs Offered by Institute", 
                 color=programs_per_institute.sort_values())
    st.plotly_chart(fig)

    # Graph 2: Seat distribution among institutes (Interactive with Plotly)
    seat_distribution = filtered_df.groupby('Institute')['Seat_Type'].count()
    st.write("### Seat Distribution Among Institutes")
    fig = px.bar(seat_distribution.sort_values(), 
                 labels={'index': 'Institute', 'value': 'Number of Seat Entries'},
                 title="Seat Distribution Across Institutes",
                 color=seat_distribution.sort_values())
    st.plotly_chart(fig)

    # Graph 3: Gender distribution in admissions (Pie chart using Plotly)
    gender_distribution = filtered_df['Gender'].value_counts()
    st.write("### Gender Distribution in Admissions")
    fig = px.pie(gender_distribution, names=gender_distribution.index, values=gender_distribution.values, 
                 title="Gender Distribution of Admissions", 
                 color=gender_distribution.index, 
                 color_discrete_map={'Male': 'lightblue', 'Female': 'lightcoral'})
    st.plotly_chart(fig)

    # Trend Analysis (Linear Regression on Seat Intake by Year)
    st.write("### Trend Analysis: Predicting Seat Intake Over the Years")
    seat_intake_by_year = filtered_df.groupby('Year')['Seat_Type'].count().reset_index()
    X = seat_intake_by_year['Year'].values.reshape(-1, 1)
    y = seat_intake_by_year['Seat_Type'].values

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seat_intake_by_year['Year'], y=seat_intake_by_year['Seat_Type'], mode='markers', name='Actual Data'))
    fig.add_trace(go.Scatter(x=seat_intake_by_year['Year'], y=predictions, mode='lines', name='Trend Line', line=dict(color='red')))
    fig.update_layout(title="Seat Intake Trend (Predicted)", xaxis_title="Year", yaxis_title="Number of Seats")
    st.plotly_chart(fig)

   #Graph 5: Changes in average closing ranks over years (Interactive with Plotly)
    avg_closing_rank_by_year = filtered_df.groupby('Year')['Closing_Rank'].mean()
    st.write("### Changes in Average Closing Ranks Over the Years")
    fig = px.line(avg_closing_rank_by_year, 
                  labels={'index': 'Year', 'Closing_Rank': 'Average Closing Rank'}, 
                  title="Average Closing Rank by Year", 
                  markers=True)
    st.plotly_chart(fig)
    # Graph 4: Seat intake among years (Interactive with Plotly)
    seat_intake_by_year = filtered_df.groupby('Year')['Seat_Type'].count()
    st.write("### Seat Intake Among Years")
    fig = px.bar(seat_intake_by_year, 
                 labels={'index': 'Year', 'value': 'Number of Seats'},
                 title="Total Seat Intake by Year", 
                 color=seat_intake_by_year)
    st.plotly_chart(fig)


    # Graph 6: Average closing rank by seat type (Interactive with Plotly)
    avg_rank_by_seat_type = filtered_df.groupby('Seat_Type')['Closing_Rank'].mean()
    st.write("### Average Closing Rank by Seat Type")
    fig = px.bar(avg_rank_by_seat_type.sort_values(), 
                 labels={'index': 'Seat Type', 'value': 'Average Closing Rank'}, 
                 title="Average Closing Rank by Seat Type", 
                 color=avg_rank_by_seat_type.sort_values())
    st.plotly_chart(fig)

    # **Download Filtered Data**
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(filtered_df)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv'
    )

   

# Tab 2: Program-Specific Analysis
with tab2:
    st.header("Program-Specific Analysis")
    
    # Sidebar filters (Program & Institute selection)
    program_options = filtered_df['Academic_Program_Name'].unique()
    institute_options = filtered_df['Institute'].unique()
    
    selected_program = st.selectbox("Select Program", program_options)
    selected_institute = st.selectbox("Select Institute", institute_options)
    
    # Filtered data based on selected program and institute
    filtered_data = filtered_df[(filtered_df['Academic_Program_Name'] == selected_program) & 
                                 (filtered_df['Institute'] == selected_institute)]
    
    # Graph: Opening Rank vs Year
    if not filtered_data.empty:
        opening_rank_by_year = filtered_data.groupby('Year')['Opening_Rank'].mean()
        st.write("### Opening Rank vs Year")
        fig = px.line(opening_rank_by_year, 
                      labels={'index': 'Year', 'Opening_Rank': 'Opening Rank'},
                      title=f"Opening Rank for {selected_program} at {selected_institute} Over the Years",
                      markers=True)
        st.plotly_chart(fig)
        
        # Graph: Closing Rank vs Year
        closing_rank_by_year = filtered_data.groupby('Year')['Closing_Rank'].mean()
        st.write("### Closing Rank vs Year")
        fig = px.line(closing_rank_by_year, 
                      labels={'index': 'Year', 'Closing_Rank': 'Closing Rank'},
                      title=f"Closing Rank for {selected_program} at {selected_institute} Over the Years",
                      markers=True)
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected program and institute combination.")

# Tab 3: Multiple Institute Comparison
with tab3:
    st.header("Multiple Institute Comparison")
    
    # Institute and Program selection for comparison
    institute_options = filtered_df['Institute'].unique()
    program_options = filtered_df['Academic_Program_Name'].unique()

    # User input for Top N Institutes/Programs
    top_n = st.slider("Select Top N Institutes/Programs to display", 1, 20, 5)

    # Dropdown for selecting Institute or Program
    comparison_type = st.selectbox("Select Comparison Type", ["Institute", "Program"])
    
    if comparison_type == "Institute":
        # Calculate average opening and closing ranks by Institute, overall
        avg_ranks_by_institute = filtered_df.groupby('Institute').agg(
            avg_opening_rank=('Opening_Rank', 'mean'),
            avg_closing_rank=('Closing_Rank', 'mean')
        ).sort_values(by='avg_opening_rank', ascending=True)
        
        # Show Top N institutes by average opening rank
        st.write(f"### Top {top_n} Institutes by Average Opening Rank")
        st.dataframe(avg_ranks_by_institute.head(top_n))

        # Show Top N institutes by average closing rank
        st.write(f"### Top {top_n} Institutes by Average Closing Rank")
        st.dataframe(avg_ranks_by_institute.sort_values(by='avg_closing_rank', ascending=True).head(top_n))
        
        # Visualization of Top N Institutes by average opening and closing ranks as Line Graphs
        st.write(f"### Visualization of Top {top_n} Institutes")
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Line plot for Top N Institutes by Opening Rank (ascending order)
        avg_ranks_by_institute.head(top_n).plot(kind='line', y='avg_opening_rank', ax=ax[0], marker='o', color='skyblue', title="Top Institutes by Opening Rank")
        ax[0].set_ylabel("Average Opening Rank")
        ax[0].set_xlabel("Institute")
        
        ax[0].tick_params(axis='x', rotation=45)
        # Line plot for Top N Institutes by Closing Rank (ascending order)
        avg_ranks_by_institute.sort_values(by='avg_closing_rank', ascending=True).head(top_n).plot(kind='line', y='avg_closing_rank', ax=ax[1], marker='o', color='salmon', title="Top Institutes by Closing Rank")
        ax[1].set_ylabel("Average Closing Rank")
        ax[1].set_xlabel("Institute")
        ax[1].tick_params(axis='x', rotation=45)
        
        st.pyplot(fig)
    
    elif comparison_type == "Program":
        # Calculate average opening and closing ranks by Program, overall
        avg_ranks_by_program = filtered_df.groupby('Academic_Program_Name').agg(
            avg_opening_rank=('Opening_Rank', 'mean'),
            avg_closing_rank=('Closing_Rank', 'mean')
        ).sort_values(by='avg_opening_rank', ascending=True)

        # Show Top N programs by average opening rank
        st.write(f"### Top {top_n} Programs by Average Opening Rank")
        st.dataframe(avg_ranks_by_program.head(top_n))

        # Show Top N programs by average closing rank
        st.write(f"### Top {top_n} Programs by Average Closing Rank")
        st.dataframe(avg_ranks_by_program.sort_values(by='avg_closing_rank', ascending=True).head(top_n))
        
        # Visualization of Top N Programs by average opening and closing ranks as Line Graphs
        st.write(f"### Visualization of Top {top_n} Programs")
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Line plot for Top N Programs by Opening Rank (ascending order)
        avg_ranks_by_program.head(top_n).plot(kind='line', y='avg_opening_rank', ax=ax[0], marker='o', color='skyblue', title="Top Programs by Opening Rank")
        ax[0].set_ylabel("Average Opening Rank")
        ax[0].set_xlabel("Program")

        ax[0].tick_params(axis='x', rotation=45)

        # Line plot for Top N Programs by Closing Rank (ascending order)
        avg_ranks_by_program.sort_values(by='avg_closing_rank', ascending=True).head(top_n).plot(kind='line', y='avg_closing_rank', ax=ax[1], marker='o', color='salmon', title="Top Programs by Closing Rank")
        ax[1].set_ylabel("Average Closing Rank")
        ax[1].set_xlabel("Program")

        ax[1].tick_params(axis='x', rotation=45)

        st.pyplot(fig)
