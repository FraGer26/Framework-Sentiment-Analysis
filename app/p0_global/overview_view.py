import streamlit as st
import plotly.graph_objects as go
from p0_global import data, general_statistics

def render_global_overview(user_data):
    st.subheader("User Activity Timeline")
    # Istogramma semplice dei post nel tempo
    daily_counts = user_data.groupby(user_data["Date"].dt.date).size()
    fig_activity = go.Figure(data=[
        go.Bar(x=daily_counts.index, y=daily_counts.values, name="Posts")
    ])
    fig_activity.update_layout(xaxis_title="Date", yaxis_title="Number of Posts", template="plotly_white")
    st.plotly_chart(fig_activity, use_container_width=True)
    
    st.subheader("Recent Posts")
    st.dataframe(user_data[["Date", "Text", "Prob_Severe_Depressed", "Prob_Moderate_Depressed"]].sort_values("Date", ascending=False).head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top Risk Posts")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### Top 5 Severely Depressed")
        top_severe = user_data.sort_values("Prob_Severe_Depressed", ascending=False).head(5)
        st.dataframe(top_severe[["Date", "Prob_Severe_Depressed", "Text"]], use_container_width=True, hide_index=True)
        
    with col_t2:
        st.markdown("#### Top 5 Moderately Depressed")
        top_moderate = user_data.sort_values("Prob_Moderate_Depressed", ascending=False).head(5)
        st.dataframe(top_moderate[["Date", "Prob_Moderate_Depressed", "Text"]], use_container_width=True, hide_index=True)
