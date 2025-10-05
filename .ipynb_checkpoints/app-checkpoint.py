import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------
# Load your processed data
# ------------------------------------
weekly_results = pd.read_csv("results/GBBO_weekly_predictions.csv")

# ------------------------------------
# Sidebar
# ------------------------------------
weekly_results['Season'] = weekly_results['Season']+1
season_list = weekly_results['Season'].unique()
selected_season = st.sidebar.selectbox("Select Season", season_list)

# ------------------------------------
# Tabs
# ------------------------------------
# tab1, tab2 = st.tabs(["📉 Elimination Risk", "🏆 Win Probabilities"])
tab_intro, tab1, tab2, tab3 = st.tabs(["Introduction", "🍰 Elimination Risk", "💻 Feature Importance", "🏆 Win Probabilities",])



import streamlit as st

with tab_intro:
    st.title("🍰 The Great British Bake Off: Elimination Forecasting 🥧")
    st.markdown("""
    ### Overview
    
    Here is a fun project exploring the **Great British Bake Off (GBBO)** results across Seasons 3-14! 🎂  
    The goal of this analysis was to **predict when a contestant will be eliminated** based on their past weekly performance, using data from 12
    seasons of GBBO (sourced from [Kaggle](https://www.kaggle.com/datasets/sarahvitvitskiy/great-british-bake-off-results-seasons-1-14)).
    
    ---

    ### Set-up
    In each episode, bakers are evaluated on three challenges — **Signature**, **Technical**, and **Showstopper** —  
    with the result for the week ranging from "Low", "Safe", "High", "Star Baker" or "Out" (eliminated).  
    I wanted to see whether past performance could **quantitatively predict** who would be eliminated next week.

    ---

    ### 🔍 Data & Features
    - Each baker’s weekly results were encoded numerically (`HIGH`, `SAFE`, `LOW`, etc.)  
    - Rolling and cumulative features were computed to capture **momentum** and **consistency**, including:  
        - Cumulative mean performance, 
        - Consecutive highs / Star Bakers, 
        - Average technical ranking, 
        - Approximate “Slope” of performance trend over past two weeks,
        - Streaks of Star Baker awards,
        etc.

    - All features were carefully designed to use only **information available before that week**,  
      preventing *data leakage*.

    ---

    ### ⚙️ Modeling
    - A **Logistic Regression** classifier was trained to estimate the probability that  
      a baker is eliminated each week.
    - The model achieved an ROC-AUC of around **0.70**, correctly predicting the eliminated baker  
      in ~33% of weeks, and ranking them in the top 3 risk group in ~73% of cases.
    - The model’s coefficients were used to interpret **what drives elimination risk** —  
      for example, poor consistency or consecutive “Low” weeks increased risk,  
      while strong average performance and streaks of “High” results were protective.

    ---

    ### 📊 Visual Dashboard
    Use the tabs above to explore:
    - **Performance Heatmaps:** Weekly elimination probabilities by baker, with elimination markers  
    - **Prediction Simulator:** Adjust how many weeks are included to see how predictions evolve  
    - **Feature Importance:** Which features most strongly predict elimination  

    ---

    ### 🧁 Key Takeaways
    - Statistical models can capture real narrative structure even in entertainment data.
    - We can (sometimes) predict who will be eliminated and who will win, i.e., GBBO eliminations aren’t random — past performance trends really matter!
    - Though we do an OK job with predictions there are still outliers, and we don't always get it right. This weeks performance matters signficantly, and we can't account for fluctuations of individual contestants performance at crunch time!
    - This project is helping me improve my skills in **feature engineering**, **model interpretability**,  
      and **data storytelling** using **Python**, **Plotly**, and **Streamlit**.

    ---

    *Project by Catriona Murray*  
    Data source: Kaggle | Built with: Python, Pandas, Scikit-learn, Streamlit, Plotly
    """)

# ------------------------------------
# TAB 1: Elimination Risk
# ------------------------------------
with tab1:
    st.subheader(f"Predicted Elimination Risk – Season {selected_season}")
    st.markdown("""
    This dashboard shows predicted elimination risk for Great British Bake Off bakers.
    - **Top 3 Table:** Highlights the bakers most likely to be eliminated in the selected week using only metrics calculated from previous weeks.
    - **Heatmap:** Shows predicted probabilities for all bakers in the season, with black dots indicating actual eliminations and gold stars indicating Star Baker awards.
    """)

    
    week_list = weekly_results[weekly_results['Season'] == selected_season]['Week'].unique()[1:]
    selected_week = st.selectbox("Select Week", week_list)

    season_data = weekly_results[(weekly_results['Season'] == selected_season)]

    season_week_data = season_data[
        # (weekly_results['Season'] == selected_season) & 
        (weekly_results['Week'] == selected_week)
    ].copy()

    # Rank by elimination probability
    season_week_data['Rank_in_Week'] = (
        season_week_data['Predicted_Elim_Prob']
        .rank(method='first', ascending=False)
        .astype(int)
    )
    top3 = season_week_data.sort_values(by="Rank_in_Week")#.nsmallest(3, 'Rank_in_Week')
    top3=top3.set_index(top3["Rank_in_Week"])

    st.table(top3[['Baker', 'Predicted_Elim_Prob', 'Actual_Eliminated']],)

    # Heatmap for elimination probabilities
    heatmap_data = (
        weekly_results[weekly_results['Season'] == selected_season]
        .pivot_table(index='Baker', columns='Week', values='Predicted_Elim_Prob')
    )

    fig_elim = px.imshow(
        heatmap_data,
        labels=dict(x="Week", y="Baker", color="Elimination Prob"),
        color_continuous_scale="plasma",#"Reds",
        aspect="auto"
    )

    eliminated = season_data[season_data["Actual_Eliminated"] == 1]
    sb = season_data[season_data["is_SB"] == 1]

    fig_elim.add_trace(
        go.Scatter(
            x=eliminated["Week"],
            y=eliminated["Baker"],
            mode="markers",
            marker=dict(color="black", size=8, symbol="circle"),
            name="Eliminated",
        )
    )

    fig_elim.add_trace(
        go.Scatter(
            x=sb["Week"],
            y=sb["Baker"],
            mode="markers",
            marker=dict(color="gold", size=8, symbol="star"),
            name="Star Baker",
        )
    )

    # Clean up layout
    fig_elim.update_layout(
        yaxis=dict(title="", autorange="reversed"),
        xaxis=dict(title="Week", dtick=1),
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0),
    )
    
    st.plotly_chart(fig_elim, use_container_width=True)

# ------------------------------------
# TAB 2: Feature Importance
# ------------------------------------
with tab2:
    st.header("Feature Importance")
    st.markdown("""
    - **Feature Importance:** Logistic regression coefficients indicating which features drive elimination predictions.
    The chart below shows the **logistic regression coefficients**.
    Positive values mean that higher feature values increase the chance
    of **elimination**, while negative values decrease it.
    """)

    # Load coefficients
    coef_df = pd.read_csv("results/GBBO_feature_importance.csv")

    # Sort by absolute importance
    coef_df["abs_coef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=True)

    fig = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu_r",
        title="Feature Importance (Logistic Regression Coefficients)",
    )

    fig.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(title=""),
        xaxis=dict(title="Coefficient", zeroline=True, zerolinecolor="black"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interpretation")

    # Feature descriptions for the GBBO model
    feature_descriptions = {
        "prior_weeks": "Number of weeks the baker has already survived.",
        "bakers_in_week": "Number of bakers still in the competition that week.",
        "cum_result_mean_prev_norm": "Normalized average performance across all prior weeks (higher = stronger).",
        "cum_sb_count_prev": "Total number of Star Baker wins prior to this week.",
        "cum_low_count_prev": "Total number of times the baker was in the bottom group.",
        "cum_tech_mean_prev": "Average technical ranking across all previous weeks.",
        "cum_hand_prev": "Number of handshakes received up to this week.",
        "last_week_result": "Result from the previous week’s performance.",
        "delta_last_two": "Change in performance between the last two weeks.",
        "last_week_tech": "Technical ranking in the previous week.",
        "consec_high_prev": "Number of consecutive weeks with a 'High' or 'Star Baker' result before this one.",
        "consec_sb_prev": "Number of consecutive Star Baker wins before this week.",
        "result_std_prev": "Variability in performance across prior weeks (consistency indicator).",
        "score_slope_prev": "Trend in performance (positive = improving over time).",
        "sb_streak": "Highest streak of Star Bakers.",
        "Gender_Binary": "1 = Female, 0 = Male.",
        "good_streak": "Highest streak of High or SB weeks",
        "last_week_sb": "Whether the baker was Star Baker last week (1=yes).",
    }


    # --- 🔹 Generate narrative for top features ---
    top_positive = (
        coef_df.sort_values("Coefficient", ascending=False)
        .head(5)[["Feature", "Coefficient"]]
        .reset_index(drop=True)
    )
    top_negative = (
        coef_df.sort_values("Coefficient", ascending=True)
        .head(5)[["Feature", "Coefficient"]]
        .reset_index(drop=True)
    )


    st.markdown("#### 🟥 Features that increase elimination risk:")
    for _, row in top_positive.iterrows():
        desc = feature_descriptions.get(row["Feature"], "No description available.")
        st.markdown(
            f"- **{row['Feature']}** (`{row['Coefficient']:.2f}`): {desc} → higher values make elimination *more likely*."
        )

    st.markdown("#### 🟦 Features that protect against elimination:")
    for _, row in top_negative.iterrows():
        desc = feature_descriptions.get(row["Feature"], "No description available.")
        st.markdown(
            f"- **{row['Feature']}** (`{row['Coefficient']:.2f}`): {desc} → higher values make elimination *less likely*."
        )



    st.markdown("""
    Overall, the model suggests that **consistent performance** and **recent success streaks**
    significantly reduce a baker’s chance of elimination, while repeated **low placements**
    and longer participation in the competition (simply surviving longer) increase it.
    """)


# ------------------------------------
# TAB 3: Win Probabilities
# ------------------------------------
with tab3:
    st.subheader(f"Estimated Win Probabilities – Season {selected_season}")
    st.markdown("""
    Each line shows the model's estimated probability that a baker will ultimately
    win, given their performance so far.  
    Probabilities are normalized each week so that all remaining bakers sum to 1.
    """)
    
    # --- 1. Compute survival and normalized win probs if not precomputed ---
    if 'P_survive_cum' not in weekly_results.columns:
        weekly_results['P_survive'] = 1 - weekly_results['Predicted_Elim_Prob']
        weekly_results['P_survive_cum'] = (
            weekly_results.groupby(['Season', 'Baker'])['P_survive'].cumprod()
        )

    weekly_results['P_win_est_weekly_norm'] = (
        weekly_results['P_survive_cum'] /
        weekly_results.groupby(['Season', 'Week'])['P_survive_cum'].transform('sum')
    )

    # --- 2. Plot win probabilities ---
    season_data = weekly_results[weekly_results['Season'] == selected_season]
    season_data['Final_Place'] = season_data['Final_Place'].astype(int)

    fig_win = px.line(
        season_data,
        x='Week',
        y='P_win_est_weekly_norm',
        color='Baker',
        title=f"Normalized Win Probability by Week – Season {selected_season}",
        labels={'P_win_est_weekly_norm': 'Win Probability'},
    )

    st.plotly_chart(fig_win, use_container_width=True)

    
    # --- 3. Highlight the final-week predicted winner ---
    final_week = season_data['Week'].max()
    final_probs = season_data[season_data['Week'] == final_week][
        ['Baker', 'P_win_est_weekly_norm', 'Final_Place']
    ].sort_values('P_win_est_weekly_norm', ascending=False)

    st.subheader("🏁 Semi-Final Win Predictions")
    st.table(final_probs.reset_index(drop=True))

    
    # Let user choose number of weeks to include
    max_week = int(weekly_results['Week'].max())
    weeks_to_include = st.slider(
        "Select number of weeks to include",
        min_value=1,
        max_value=max_week,
        value=max_week,  # default: show all
        step=1
    )

    # Filter based on selected number of weeks
    filtered = weekly_results[weekly_results['Week'] <= weeks_to_include]
    filtered = filtered[filtered['Season'] == selected_season]
    # st.dataframe(
    #     filtered[['Week', 'Baker', 'Predicted_Elim_Prob', 'Actual_Eliminated']]
    #     .sort_values(['Week', 'Predicted_Elim_Prob'], ascending=[True, False])
    #     .reset_index(drop=True)
    # )
    
    # Show elimination probabilities
    st.subheader(f"Estimated Win Probabilities, Un-normalized (Weeks 1–{weeks_to_include})")

    # Optional: chart
    st.line_chart(
        filtered.pivot_table(
            index='Week', columns='Baker', values='P_win_est_weekly'
        )
    )

    # --- 3. Highlight the final-week predicted winner ---
    # final_week = season_data['Week'].max()
    final_week = weeks_to_include
    final_probs = season_data[season_data['Week'] == final_week][
        ['Baker', 'P_win_est_weekly_norm', 'Final_Place']
    ].sort_values('P_win_est_weekly_norm', ascending=False)


    st.subheader(f"Win Predictions at Week {weeks_to_include}")
    st.table(final_probs.reset_index(drop=True))

    

