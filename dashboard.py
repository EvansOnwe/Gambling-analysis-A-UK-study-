import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
import joblib

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gambling Participation & Severity – UK",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS (gray background, soft cards, academic style)
# ---------------------------------------------------------




st.markdown("""
<style>
html {
    background-color: #f0f0f0 !important; /* Gray background for the whole page */
}

body {
    background-color: #f0f0f0 !important; /* Ensure dashboard background is gray */
    color: #000000 !important; /* Black text */
}

.main-title {
    font-size: 36px !important;
    font-weight: 700 !important;
    color: #3cfff !important;
    margin-bottom: 0px !important;
}

.subtitle {
    font-size: 16px;
    color: #000000 !important; /* Black text for subtitles */
    margin-bottom: 30px;
}

.metric-card {
    background-color: #ffffff !important; /* White background for metric cards */
    color: #000000 !important; /* Black text for metric cards */
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
    height: 130px;
}

.metric-title {
    font-size: 14px;
}

.metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #0F4C5C;
    margin-top: 8px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #000000 !important; /* Black text for section titles */
    margin-top: 40px;
    margin-bottom: 10px;
}

.caption {
    font-size: 13px;
    color: #000000 !important; /* Black text for captions */
    margin-bottom: 15px;
}

.plot-card {
    background-color: #ffffff !important; /* White background for plot cards */
    color: #000000 !important; /* Black text for plot cards */
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
    height: 300px; /* Adjust height for plots */
}

/* Sidebar container padding */
section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* Radio / navigation item spacing */
div[role="radiogroup"] > label {
    margin-bottom: 1rem;
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
}

/* Improve text readability */
div[role="radiogroup"] > label span {
    font-size: 0.95rem;
    letter-spacing: 0.3px;
}
            
</style>
""", unsafe_allow_html=True)



# Load the model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Admin\Documents\gambling project\models\best_model\best_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file not found: {model_path}")
        return None

loaded_model = load_model()


# Load data
@st.cache_data
def load_base_data():
    file_path = r"C:\Users\Admin\Documents\gambling project\outputs\combined_data2.pkl"
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        st.error(f"File not found: {file_path}")
        return None

combined_data2 = load_base_data()
df = combined_data2.copy()


#load X_test_scaled
@st.cache_data
def load_X_test_scaled():
    file_path = r"C:\Users\Admin\Documents\gambling project\dataset\processed\X_test_scaled.pkl"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        st.error(f"File not found: {file_path}")
        return None
    
X_test_scaled = load_X_test_scaled()


@st.cache_data
def load_X():
    file_path = r"C:\Users\Admin\Documents\gambling project\dataset\processed\X.pkl"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        st.error(f"File not found: {file_path}")
        return None

X = load_X()

# ---------------------------------------------------------
# Sidebar with Logo and Navigation
# st.sidebar.image(
#     r"C:\Users\Admin\Documents\gambling project\Images\logo.png",
#     use_container_width=True
# )

# st.sidebar.title("Dashboard Sections")

section = st.sidebar.radio(
    "NAVIGATION",
    ["OVERVIEW", "EXPLORATORY DATA ANALYSIS", "MODEL EVALUATION", "SHAP EXPLAINABILITY AND FEATURE IMPORTANCE", "RESEARCH QUESTIONS"]
)

st.markdown(
    "<div class='main-title'>The impact of gambling among UK Adults: A descriptive and machine learning approach</div>",
    unsafe_allow_html=True
)

if section == "OVERVIEW":
    # ---------------------------------------------------------
    # Title with "View Data" Toggle Button
    # ---------------------------------------------------------

    if "show_data" not in st.session_state:
        st.session_state["show_data"] = False

    if st.button("View Data"):
        st.session_state["show_data"] = not st.session_state["show_data"]

    if st.session_state["show_data"]:
        file_path = r"C:\Users\Admin\Documents\gambling project\outputs\combined_data2.pkl"
        if os.path.exists(file_path):
            combined_data2 = pd.read_pickle(file_path)
            # Center the dataframe using Streamlit's layout system
            st.columns([1, 6, 1])  # Add padding columns to center the dataframe
            st.dataframe(combined_data2, use_container_width=True)
        else:
            st.error(f"File not found: {file_path}")

    # ---------------------------------------------------------
    # Key Metrics Row (5 cards)
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Severe Gambling Probability</div>
            <div class="metric-value">0.72%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Calculate Participation Rate
        if combined_data2 is not None:
            participation_rate = (
                combined_data2["Gambling_Participation"].mean() * 100
            ).round(2)
            participation_rate_display = f"{participation_rate}%"
        else:
            participation_rate_display = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Participation Rate</div>
            <div class="metric-value">{participation_rate_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Replace "Total Participants" with the count from combined_data
        if combined_data2 is not None:
            total_participants = combined_data2["Survey_Year"].count()
        else:
            total_participants = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Participants</div>
            <div class="metric-value">{total_participants}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Calculate Male vs Female Gamblers
        if combined_data2 is not None:
            # Ensure the column name matches the dataset
            if "Sex" in combined_data2.columns and "Gambling_Participation" in combined_data2.columns:
                df = combined_data2[combined_data2["Gambling_Participation"] == 1]
                sex_counts = df["Sex"].value_counts(dropna=False)
                male_count = sex_counts.get("Male", 0)
                female_count = sex_counts.get("Female", 0)
                male_vs_female = f"{male_count} M / {female_count} F"
            else:
                male_vs_female = "Column Missing"
        else:
            male_vs_female = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Number of Male and Female Gamblers</div>
            <div class="metric-value">{male_vs_female}</div>
        </div>
        """, unsafe_allow_html=True)



elif section == "EXPLORATORY DATA ANALYSIS":
    st.markdown("---")
    st.markdown("<H3>EXPLORATORY DATA ANALYSIS</H3>", unsafe_allow_html=True)

    with st.expander("Summary Statistics"):
        df = combined_data2.copy()
        st.dataframe(df.describe(), use_container_width=True)


    
    with st.expander("Distribution of Key Variables"):

        # Load dataset
        df = combined_data2.copy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        #Distribution of PGSI Scores
        axes[0, 0].hist(df["pgsi_Score"].dropna(), bins=20,  color="#00A8C2")
        axes[0, 0].set_title("Distribution of PGSI Scores")
        axes[0, 0].set_xlabel("PGSI Score")
        axes[0, 0].set_ylabel("Frequency")

        #Distribution of Gambling Severity (PGSI Categories)
        pgsi_dist = df["pgsi_category"].value_counts(dropna=False)
        bars = axes[0, 1].bar(pgsi_dist.index.astype(str), pgsi_dist.values,  color="#00A8C2")
        axes[0, 1].set_title("Distribution of Gambling Severity (PGSI Categories)")
        axes[0, 1].set_xlabel("PGSI Category")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        # Dstribution of Gambling Participation
        gambled_counts = df["Gambling_Participation"].value_counts().sort_index()
        bars = axes[1, 0].bar(gambled_counts.index.astype(str), gambled_counts.values,  color="#00A8C2")
        axes[1, 0].set_title("Distribution of Gambling Participation")
        axes[1, 0].set_xlabel("Gambled (0 = No, 1 = Yes)")
        axes[1, 0].set_ylabel("Count")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        #Online vs Offline Gambling Distribution
        online_counts = df["Online_Gambling"].value_counts(dropna=False)
        bars = axes[1, 1].bar(online_counts.index.astype(str), online_counts.values,  color="#00A8C2")
        axes[1, 1].set_title("Online vs Offline Gambling Distribution")
        axes[1, 1].set_xlabel("Gambling Mode")
        axes[1, 1].set_ylabel("Number of Respondents")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # Final layout
        plt.tight_layout()
        st.pyplot(plt.gcf()) 



    with st.expander("Gambling participation across survey years"):

            # Load dataset
            df = combined_data2.copy()

            # Calculate participation by year & sex
            sex_participation = (
                df.groupby(["Survey_Year", "Sex"])["Gambling_Participation"]
                .mean()
                .unstack() * 100
            )

            # Overall participation (for line plot)
            overall_participation = (
                df.groupby("Survey_Year")["Gambling_Participation"]
                .mean() * 100
            )

            years = sex_participation.index.astype(str)

            # Create 1x2 canvas
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            plt.subplots_adjust(wspace=0.35)

            # LEFT: Grouped Bar Plot (Male vs Female)
            bar_width = 0.35
            x = np.arange(len(years))

            male_vals = sex_participation["Male"]
            female_vals = sex_participation["Female"]

            color="#00A8C2"
            bars_male = axes[0].bar(x - bar_width/2, male_vals, width=bar_width, label="Male", color="#00A8C2")
            bars_female = axes[0].bar(x + bar_width/2, female_vals, width=bar_width, label="Female", color="#A18F5E")

            axes[0].set_title("Gambling Participation by Survey Year and Sex")
            axes[0].set_xlabel("Survey Year")
            axes[0].set_ylabel("Percentage of Gamblers (%)")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(years)
            axes[0].legend()

            # Add value labels
            for bars in [bars_male, bars_female]:
                for bar in bars:
                    height = bar.get_height()
                    axes[0].text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )

            # RIGHT: Line Plot (Overall Trend)
            axes[1].plot(years, overall_participation.values, marker="o")
            axes[1].set_title("Overall Gambling Participation Trend")
            axes[1].set_xlabel("Survey Year")
            axes[1].set_ylabel("Percentage of Gamblers (%)")

            # Add value labels
            for x_val, y_val in zip(years, overall_participation.values):
                axes[1].text(x_val, y_val, f"{y_val:.1f}%", ha="center", va="bottom")

            plt.tight_layout()
            st.pyplot(plt.gcf())



    with st.expander("Distribution of Gamblers across demographics varibles"):

        df = df[combined_data2.copy()["Gambling_Participation"] == 1]
        # Create 2x2 canvas
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Age Group Distribution
        age_counts = df["Age_Group"].value_counts(dropna=False).sort_index()
        bars = axes[0, 0].bar(age_counts.index.astype(str), age_counts.values, color="#00A8C2")
        axes[0, 0].set_title("Age Group Distribution")
        axes[0, 0].set_xlabel("Age Group")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        #Sex Distribution
        sex_counts = df["Sex"].value_counts(dropna=False)
        bars = axes[0, 1].bar(sex_counts.index.astype(str), sex_counts.values, color="#00A8C2")
        axes[0, 1].set_title("Sex Distribution")
        axes[0, 1].set_xlabel("Sex")
        axes[0, 1].set_ylabel("Count")

        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        #Educational Attainment Distribution
        edu_counts = df["Education_Level"].value_counts(dropna=False)
        bars = axes[1, 0].bar(edu_counts.index.astype(str), edu_counts.values, color="#00A8C2")
        axes[1, 0].set_title("Educational Attainment Distribution")
        axes[1, 0].set_xlabel("Education Level")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )
        # Regional Distribution
        region_counts = df["Region"].value_counts(dropna=False)
        bars = axes[1, 1].bar(region_counts.index.astype(str), region_counts.values, color="#00A8C2")
        axes[1, 1].set_title("Regional Distribution")
        axes[1, 1].set_xlabel("Region")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        # Final layout
        plt.tight_layout()
        st.pyplot(plt.gcf())



    with st.expander("Distribution of Gambling Participation Patterns"):

        # Load dataset
        df = combined_data2.copy()

        # Create 1x3 canvas
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        #Proportion of Gamblers vs Non-Gamblers
        gambled_counts = df["Gambling_Participation"].value_counts().sort_index()
        bars = axes[0].bar(gambled_counts.index.astype(str), gambled_counts.values, color="#00A8C2")
        axes[0].set_title("Gamblers vs Non-Gamblers")
        axes[0].set_xlabel("Gambling Participation (0 = No, 1 = Yes)")
        axes[0].set_ylabel("Count")

        for bar in bars:
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        #Online vs Offline Gambling Participation
        online_counts = df["Online_Gambling"].value_counts(dropna=False)
        bars = axes[1].bar(online_counts.index.astype(str), online_counts.values, color="#00A8C2")
        axes[1].set_title("Online vs Offline Gambling")
        axes[1].set_xlabel("Gambling Mode")
        axes[1].set_ylabel("Count")

        for bar in bars:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom"
            )

        #Gambling Participation Across Survey Years
        year_participation = (
            df.groupby("Survey_Year")["Gambling_Participation"]
            .mean() * 100
        )

        bars = axes[2].bar(
            year_participation.index.astype(str),
            year_participation.values,
            color="#00A8C2"
        )
        axes[2].set_title("Gambling Participation Across Survey Years")
        axes[2].set_xlabel("Survey Year")
        axes[2].set_ylabel("Percentage of Gamblers (%)")

        for bar in bars:
            height = bar.get_height()
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom"
            )

        # Final layout
        plt.tight_layout()
        st.pyplot(plt.gcf())



    with st.expander("correlation analysis (pgsi score, pgsi grouped and gambling participation)"):

    
        df = combined_data2.copy()

        # Map PGSI category to ordered numeric scale
        pgsi_map = {
            "Non-problem / Non-gambler": 0,
            "Low-risk gambler": 1,
            "Moderate-risk gambler": 2,
            "Problem gambler": 3
        }

        df["PGSI_Category_Num"] = df["pgsi_category"].map(pgsi_map)

        # Select variables
        corr_df = df[
            ["pgsi_Score", "PGSI_Category_Num", "Gambling_Participation"]
        ].apply(pd.to_numeric, errors="coerce")

        # Compute correlation matrix
        corr_matrix = corr_df.corr()

        # Plot heatmap with desired colour scheme
        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            corr_matrix,
            cmap="coolwarm", 
            vmin=-1,
            vmax=1,
            aspect="equal"
        )
        plt.colorbar(im)

        plt.xticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns,
            rotation=45
        )
        plt.yticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns
        )

        # Add numeric values inside each cell
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                plt.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10
                )

        plt.title(
            "Correlation Matrix: PGSI Score, PGSI Category and Gambling Participation"
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())



    with st.expander("correlation of categorical variables and PGSI Score"):

        df = combined_data2.copy()

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            corr_matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="equal"
        )
        plt.colorbar(im)

        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

        # Annotate values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                plt.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9
                )

        plt.title("Correlation Matrix: PGSI Score and Demographic Variables")
        plt.tight_layout()
        st.pyplot(plt.gcf())

        



elif section == "MODEL EVALUATION":
    st.markdown("---")
    st.markdown("<H3>MODEL EVALUATION</H3>", unsafe_allow_html=True)

    with st.expander("Model Performance Metrics"):

        models = ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"]

        f1_scores = [0.000, 0.136, 0.000, 0.133]
        precision_scores = [0.000, 0.429, 0.000, 0.375]
        recall_scores = [0.000, 0.081, 0.000, 0.081]
        roc_auc_scores = [0.888, 0.690, 0.695, 0.867]
        accuracy_scores = [0.993, 0.993, 0.993, 0.992]

        # Bar positions
        x = np.arange(len(models))
        width = 0.15

        fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)

        # Plot bars
        bars_f1 = ax.bar(x - 2*width, f1_scores, width, label="F1 Score", color="#000000")
        bars_precision = ax.bar(x - width, precision_scores, width, label="Precision", color="#6A5ACD")
        bars_recall = ax.bar(x, recall_scores, width, label="Recall", color="#1D8800")
        bars_roc = ax.bar(x + width, roc_auc_scores, width, label="ROC-AUC", color="#D68800")
        bars_acc = ax.bar(x + 2*width, accuracy_scores, width, label="Accuracy", color="#00A8C2")

        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        add_labels(bars_f1)
        add_labels(bars_precision)
        add_labels(bars_recall)
        add_labels(bars_roc)
        add_labels(bars_acc)

        # Axes & formatting
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Model Performance Evaluation", fontsize=14)

        # Legend outside plot
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=10
        )

        plt.tight_layout()
        st.pyplot(plt.gcf())


        st.markdown("---")

        st.markdown("Best model vs Improved model")

        models = ["Original GB", "Tuned GB"]

        f1 = [0.133, 0.093]
        roc_auc = [0.867, 0.875]
        accuracy = [0.992, 0.992]

        # Bar setup
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

        # 🔁 Order: F1 → ROC-AUC → Accuracy
        bars_f1 = ax.bar(x - width, f1, width, label="F1-score", color="#00A8C2")
        bars_auc = ax.bar(x, roc_auc, width, label="ROC-AUC", color="#FFA500")
        bars_acc = ax.bar(x + width, accuracy, width, label="Accuracy", color="#A18F5E")

        # Value labels
        for bars in [bars_f1, bars_auc, bars_acc]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

        # Axes & title
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("Metric Value")
        ax.set_title("Performance Comparison: Original vs Tuned Gradient Boosting Model")

        # Legend
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )

        plt.tight_layout()
        st.pyplot(plt.gcf())





elif section == "SHAP EXPLAINABILITY AND FEATURE IMPORTANCE":
    st.markdown("---")
    st.markdown("<H3>SHAP EXPLAINABILITY AND FEATURE IMPORTANCE</H3>", unsafe_allow_html=True)

    with st.expander("SHAP Explainability and feature importance plots"):
                #Create SHAP explainer
        explainer = shap.TreeExplainer(loaded_model)

        #Compute SHAP values for test data
        shap_values = explainer.shap_values(X_test_scaled)

        shap.summary_plot(
            shap_values,
            X_test_scaled,
            feature_names=X.columns
        )

        st.pyplot(plt.gcf())

        st.markdown("---")
        st.markdown("Feature Importance Plot")

        # Convert SHAP values to numpy
        if hasattr(shap_values, "values"):
            shap_array = shap_values.values
        else:
            shap_array = shap_values

        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_array).mean(axis=0)

        # Create DataFrame
        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": mean_abs_shap
        })

        # Sort and keep top N features (reduce clutter)
        TOP_N = 20
        shap_importance = (
            shap_importance
            .sort_values("mean_abs_shap", ascending=False)
            .head(TOP_N)
        )


        fig, ax = plt.subplots(figsize=(9, 10))  # wider figure → longer bars

        bars = ax.barh(
            shap_importance["feature"],
            shap_importance["mean_abs_shap"],
            color="#00D941FF"
        )

        # Add value labels at end of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.005,  # small offset
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=8
            )

        # Axis formatting
        ax.invert_yaxis()
        ax.set_xlabel(
            "mean(SHAP value) – average impact on model output",
            fontsize=8
        )

        ax.set_title(
            "Top Predictors of Severe Gambling (SHAP)",
            fontsize=8
        )

        # Tick label size (same everywhere)
        ax.tick_params(axis="both", labelsize=8)

        plt.tight_layout()
        st.pyplot(plt.gcf())






elif section == "RESEARCH QUESTIONS":
    st.markdown("---")
    st.markdown("<H3>RESEARCH QUESTIONS</H3>", unsafe_allow_html=True)

    st.markdown("---")


    with st.expander("WHAT IS THE TREND IN GAMBLING PARTICIPATION AND GAMBLING SEVERITY AMONG UK ADULTS ACROSS SURVEY YEARS?"):
        st.markdown("<h3>Trend in gambling participation and severity</h3>", unsafe_allow_html=True)
        df = combined_data2.copy()

        # Ensure correct types
        df["pgsi_Score"] = pd.to_numeric(df["pgsi_Score"], errors="coerce")
        df["Gambling_Participation"] = pd.to_numeric(df["Gambling_Participation"], errors="coerce")
        df["Survey_Year"] = pd.to_numeric(df["Survey_Year"], errors="coerce")

        # Drop invalid rows
        df = df.dropna(subset=["Survey_Year", "Gambling_Participation"])
        df["Gambling_Participation"] = df["Gambling_Participation"].astype(int)

        # Create trend table
        trend = (
            df.groupby("Survey_Year")
            .apply(lambda g: pd.Series({
                "participation_rate": (g["Gambling_Participation"] == 1).mean(),
                "mean_pgsi_all": g["pgsi_Score"].mean(skipna=True),
                "mean_pgsi_gamblers": g.loc[
                    g["Gambling_Participation"] == 1, "pgsi_Score"
                ].mean(skipna=True)
            }))
            .reset_index()
            .sort_values("Survey_Year")
        )

        # Create 1×2 canvas
        fig, axes = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)

        # Transparent background
        fig.patch.set_alpha(0)
        for ax in axes:
            ax.set_facecolor("none")

        # Space between plots
        plt.subplots_adjust(wspace=0.35)

        # COMMON VISIBILITY FIXES (applied to both axes)
        for ax in axes:
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_alpha(0.35)

            ax.grid(True, color="white", alpha=0.12)
            ax.margins(y=0.15)

        # LEFT: Participation trend
        axes[0].plot(
            trend["Survey_Year"],
            trend["participation_rate"],
            marker="o",
            linewidth=2,
            color="#3BA9FF"
        )

        for x, y in zip(trend["Survey_Year"], trend["participation_rate"]):
            axes[0].annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="white",
                fontweight="bold"
            )

        axes[0].set_title("Trend in Gambling Participation by Survey Year", fontsize=11)
        axes[0].set_xlabel("Survey Year")
        axes[0].set_ylabel("Participation Rate")
        axes[0].set_xticks(trend["Survey_Year"])

        # RIGHT: Severity trend
        axes[1].plot(
            trend["Survey_Year"],
            trend["mean_pgsi_all"],
            marker="o",
            linewidth=2,
            label="Mean PGSI (All)",
            color="#00B3FF"
        )

        axes[1].plot(
            trend["Survey_Year"],
            trend["mean_pgsi_gamblers"],
            marker="o",
            linewidth=2,
            label="Mean PGSI (Gamblers only)",
            color="#E0C36A"
        )

        # Value labels (high contrast, no clipping)

        for x, y in zip(trend["Survey_Year"], trend["mean_pgsi_all"]):
            axes[1].annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=9,
                color="#00E5FF",
                fontweight="bold"
            )

        for x, y in zip(trend["Survey_Year"], trend["mean_pgsi_gamblers"]):
            axes[1].annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="#FFD966",
                fontweight="bold"
            )

        axes[1].set_title("Trend in Gambling Severity (Mean PGSI) by Survey Year", fontsize=11)
        axes[1].set_xlabel("Survey Year")
        axes[1].set_ylabel("Mean PGSI Score")
        axes[1].set_xticks(trend["Survey_Year"])

        # Legend (visible on dark bg)
        axes[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=9,
            frameon=False,
            labelcolor="white"
        )
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
    with st.expander("Gambling vulnerability across various demographics groups (gender, age, education level, and region) among UK adults"):
        
        df = combined_data2.copy()

        # Ensure PGSI score is numeric
        df["pgsi_Score"] = pd.to_numeric(df["pgsi_Score"], errors="coerce")

        # Calculate mean PGSI score by gender
        gender_pgsi = df.groupby("Sex")["pgsi_Score"].mean()

        # ---------------------------------------------------------
        # Create figure (slightly bigger)
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))

        # Transparent background
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # ---------------------------------------------------------
        # Bar plot
        # ---------------------------------------------------------
        bars = ax.bar(
            gender_pgsi.index.astype(str),
            gender_pgsi.values,
            color="#00A8C2"
        )

        # ---------------------------------------------------------
        # Titles & labels (VISIBLE on dark bg)
        # ---------------------------------------------------------
        ax.set_title(
            "Gender More Vulnerable to Gambling",
            fontsize=13,
            color="white"
        )
        ax.set_xlabel("Gender", fontsize=11, color="white")
        ax.set_ylabel("Mean PGSI Score", fontsize=11, color="white")

        # ---------------------------------------------------------
        # Axis ticks & spines (VISIBILITY FIX)
        # ---------------------------------------------------------
        ax.tick_params(axis="x", colors="white", labelsize=10)
        ax.tick_params(axis="y", colors="white", labelsize=10)

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.35)

        # Subtle grid
        ax.grid(axis="y", color="white", alpha=0.12)

        # Add padding so bars/labels don’t clip
        ax.margins(y=0.15)

        # ---------------------------------------------------------
        # Value labels (HIGH CONTRAST)
        # ---------------------------------------------------------
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color="white",
                fontweight="bold"
            )

        plt.tight_layout()
        st.pyplot(plt.gcf())

        st.markdown("---")




        df = combined_data2.copy()

        # Ensure PGSI numeric
        df["pgsi_Score"] = pd.to_numeric(df["pgsi_Score"], errors="coerce")

        # Keep only male and female
        df = df[df["Sex"].isin(["Male", "Female"])]

        # Compute mean PGSI by age × sex
        age_sex_pgsi = (
            df.groupby(["Age_Group", "Sex"])["pgsi_Score"]
            .mean()
            .unstack(fill_value=0)
        )

        # Compute total mean PGSI for sorting & labels
        age_sex_pgsi["Total"] = age_sex_pgsi["Male"] + age_sex_pgsi["Female"]

        # Sort age groups by total severity
        age_sex_pgsi = age_sex_pgsi.sort_values("Total", ascending=False)

        # Extract values
        ages = age_sex_pgsi.index
        male_vals = age_sex_pgsi["Male"]
        female_vals = age_sex_pgsi["Female"]

        # Create figure (bigger and transparent)
        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # Stacked bars
        bars_male = ax.bar(
            ages,
            male_vals,
            label="Male",
            color="#00A8C2"
        )

        bars_female = ax.bar(
            ages,
            female_vals,
            bottom=male_vals,
            label="Female",
            color="#A18F5E"
        )

        # Axis visibility (dark background fix)
        ax.set_title(
            "Age Group More Vulnerable to Gambling",
            fontsize=13,
            color="white"
        )
        ax.set_xlabel("Age Group", fontsize=11, color="white")
        ax.set_ylabel("Mean PGSI Score", fontsize=11, color="white")

        ax.tick_params(axis="x", colors="white", labelsize=10)
        ax.tick_params(axis="y", colors="white", labelsize=10)

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.35)

        ax.grid(axis="y", color="white", alpha=0.12)
        ax.margins(y=0.18)

        plt.xticks(rotation=45)


        # Value labels inside bars

        for bar in bars_male:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    (bar.get_x() + bar.get_width() / 2, height / 2),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )

        for bar_m, bar_f in zip(bars_male, bars_female):
            f_height = bar_f.get_height()
            if f_height > 0:
                ax.annotate(
                    f"{f_height:.3f}",
                    (
                        bar_f.get_x() + bar_f.get_width() / 2,
                        bar_m.get_height() + f_height / 2
                    ),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )

        # Total value labels on top
        for i, total in enumerate(age_sex_pgsi["Total"]):
            ax.annotate(
                f"{total:.3f}",
                (i, total),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="white",
                fontweight="bold"
            )

        # Legend (visible on dark bg)
        ax.legend(
            loc="upper right",
            fontsize=10,
            frameon=False,
            labelcolor="white"
        )

        st.pyplot(fig)

        st.markdown("---")





        # Load data
        df = combined_data2.copy()

        # Ensure PGSI numeric
        df["pgsi_Score"] = pd.to_numeric(df["pgsi_Score"], errors="coerce")

        # Keep only male and female
        df = df[df["Sex"].isin(["Male", "Female"])]

        # Compute mean PGSI by education × sex
        edu_sex_pgsi = (
            df.groupby(["Education_Level", "Sex"])["pgsi_Score"]
            .mean()
            .unstack(fill_value=0)
        )

        # Compute total mean PGSI
        edu_sex_pgsi["Total"] = edu_sex_pgsi["Male"] + edu_sex_pgsi["Female"]

        # Sort by total severity
        edu_sex_pgsi = edu_sex_pgsi.sort_values("Total", ascending=False)

        # Extract values
        edu_levels = edu_sex_pgsi.index
        male_vals = edu_sex_pgsi["Male"]
        female_vals = edu_sex_pgsi["Female"]

        # Create figure (bigger + transparent)
        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # Stacked bars
        bars_male = ax.bar(
            edu_levels,
            male_vals,
            label="Male",
            color="#00A8C2"
        )

        bars_female = ax.bar(
            edu_levels,
            female_vals,
            bottom=male_vals,
            label="Female",
            color="#A18F5E"
        )


        # Axis visibility (dark background fix)
        ax.set_title(
            "Educational Groups More Vulnerable to Gambling (Mean PGSI)",
            fontsize=13,
            color="white"
        )
        ax.set_xlabel("Educational Attainment", fontsize=11, color="white")
        ax.set_ylabel("Mean PGSI Score", fontsize=11, color="white")

        ax.tick_params(axis="x", colors="white", labelsize=10)
        ax.tick_params(axis="y", colors="white", labelsize=10)

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.35)

        ax.grid(axis="y", color="white", alpha=0.12)
        ax.margins(y=0.18)

        plt.xticks(rotation=45, ha="right")


        # Value labels inside bars

        for bar in bars_male:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    (bar.get_x() + bar.get_width() / 2, height / 2),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )

        for bar_m, bar_f in zip(bars_male, bars_female):
            f_height = bar_f.get_height()
            if f_height > 0:
                ax.annotate(
                    f"{f_height:.3f}",
                    (
                        bar_f.get_x() + bar_f.get_width() / 2,
                        bar_m.get_height() + f_height / 2
                    ),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )

        # Total value labels on top
        for i, total in enumerate(edu_sex_pgsi["Total"]):
            ax.annotate(
                f"{total:.3f}",
                (i, total),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="white",
                fontweight="bold"
            )

        # Legend (visible on dark bg)
        ax.legend(
            loc="upper right",
            fontsize=10,
            frameon=False,
            labelcolor="white"
        )

        st.pyplot(plt.gcf())

        st.markdown("---")



        # Load data
        df = combined_data2.copy()

        # Ensure numeric PGSI
        df["pgsi_Score"] = pd.to_numeric(df["pgsi_Score"], errors="coerce")

        # Keep only male and female
        df = df[df["Sex"].isin(["Male", "Female"])]

        # Compute mean PGSI by region × sex
        mean_pgsi = (
            df.groupby(["Region", "Sex"])["pgsi_Score"]
            .mean()
            .unstack(fill_value=0)
        )

        # Compute total mean PGSI
        mean_pgsi["Total"] = mean_pgsi["Male"] + mean_pgsi["Female"]

        # Sort by total mean severity
        mean_pgsi = mean_pgsi.sort_values("Total", ascending=False)

        # Create figure (bigger and  transparent)
        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        regions = mean_pgsi.index
        male_vals = mean_pgsi["Male"]
        female_vals = mean_pgsi["Female"]

        # Stacked bars
        bars_male = ax.bar(
            regions,
            male_vals,
            label="Male",
            color="#00A8C2"
        )

        bars_female = ax.bar(
            regions,
            female_vals,
            bottom=male_vals,
            label="Female",
            color="#A18F5E"
        )

        # Axis visibility (dark background fix)
        ax.set_title(
            "Regions More Vulnerable to Gambling (Mean PGSI by Region and Sex)",
            fontsize=13,
            color="white"
        )
        ax.set_xlabel("Region", fontsize=11, color="white")
        ax.set_ylabel("Mean PGSI Score", fontsize=11, color="white")

        ax.tick_params(axis="x", colors="white", labelsize=10)
        ax.tick_params(axis="y", colors="white", labelsize=10)

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.35)

        ax.grid(axis="y", color="white", alpha=0.12)
        ax.margins(y=0.18)

        plt.xticks(rotation=45, ha="right")

        # Value labels inside bars
        for bar in bars_male:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    (bar.get_x() + bar.get_width() / 2, height / 2),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )

        for bar_m, bar_f in zip(bars_male, bars_female):
            f_height = bar_f.get_height()
            if f_height > 0:
                ax.annotate(
                    f"{f_height:.3f}",
                    (
                        bar_f.get_x() + bar_f.get_width() / 2,
                        bar_m.get_height() + f_height / 2
                    ),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )

        # Total value labels on top
        for i, total in enumerate(mean_pgsi["Total"]):
            ax.annotate(
                f"{total:.3f}",
                (i, total),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="white",
                fontweight="bold"
            )

        ax.legend(
            loc="upper right",
            fontsize=10,
            frameon=False,
            labelcolor="white"
        )

        st.pyplot(plt.gcf())

    with st.expander(" Research question 3 and 4"):
        st.markdown("Research question 3 is answered in the overview section on the dashboard")
        st.markdown("SHAp and feature importnace answers the fourth research question")
      