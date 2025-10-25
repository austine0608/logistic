
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.figure_factory as ff

# -----------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------
st.set_page_config(page_title="Lotus-Gold Consulting", layout="wide")


st.header('Lotus-Gold Consulting')
st.markdown('---')
st.image('./4.jpg')
st.markdown('---')

st.title("Logistic Regression")

st.write("""
Welcome! This app uses **Logistic Regression** to predict whether a person is at risk of **heart disease** 
based on medical attributes such as age, cholesterol, blood pressure, etc.
""")

# -----------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dataset", "⚙️ Model Training", "🔮 Prediction"])

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
# Load dataset
df = pd.read_csv("Heart.csv")

# Display data preview
with st.expander("Heart Dataset"):
    st.dataframe(df)
# -----------------------------------------------------
# 1️⃣ Dataset Overview Page
# -----------------------------------------------------
if page == "Dataset":
    st.header("Dataset Overview")

    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap (Interactive)")
    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    selected_feature = st.selectbox("Select Feature to Visualize", numeric_cols)
    fig_hist = px.histogram(df, x=selected_feature, nbins=40, color_discrete_sequence=["teal"],
                            title=f"Distribution of {selected_feature}", marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------
# 2️Model Training Page
# -----------------------------------------------------
elif page == " Model Training":
    st.header("Model Training and Evaluation")

    target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    feature_cols = st.multiselect(
        "Select Feature Columns (X)",
        [c for c in df.columns if c != target_col],
        default=[c for c in df.columns if c != target_col]
    )

    if len(feature_cols) > 0:
        X = df[feature_cols]
        y = df[target_col]

        test_size = st.slider("Test size (proportion for testing)", 0.1, 0.5, 0.3)
        random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42, step=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -----------------------------------------------------
        # Model Performance
        # -----------------------------------------------------
        st.subheader("Model Evaluation")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc*100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = ["No Disease", "Disease"] if len(np.unique(y)) == 2 else [str(i) for i in np.unique(y)]
        fig_cm = ff.create_annotated_heatmap(
            cm, x=labels, y=labels, colorscale="Teal", showscale=True
        )
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Probability Visualization
        st.subheader("Prediction Probabilities (Interactive)")
        probs = model.predict_proba(X_test)
        prob_df = pd.DataFrame(probs, columns=[f"Prob_{label}" for label in np.unique(y)])
        prob_df["Actual"] = y_test.reset_index(drop=True)
        prob_df["Predicted"] = y_pred

        fig_prob = px.scatter(
            prob_df,
            x=prob_df.columns[0],
            y=prob_df.columns[1] if len(prob_df.columns) > 1 else prob_df.columns[0],
            color=prob_df["Actual"].astype(str),
            symbol=prob_df["Predicted"].astype(str),
            title="Prediction Probability Scatter Plot",
            hover_data=["Actual", "Predicted"]
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Save model for next section
        st.session_state["model"] = model
        st.session_state["features"] = feature_cols

    else:
        st.warning("Please select at least one feature column to continue.")

# -----------------------------------------------------
# 3️Prediction Page
# -----------------------------------------------------
elif page == "Prediction":
    st.header("Make a New Prediction")

    if "model" not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' section.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["features"]

        user_input = {}
        for feature in feature_cols:
            val = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
            user_input[feature] = val

        if st.button("Predict Heart Disease Risk"):
            user_df = pd.DataFrame([user_input])
            prediction = model.predict(user_df)[0]
            prob = model.predict_proba(user_df)[0]

            if prediction == 1:
                st.error(f"High Risk of Heart Disease! (Probability: {prob[1]*100:.2f}%)")
            else:
                st.success(f"Low Risk of Heart Disease (Probability: {prob[0]*100:.2f}%)")

        st.info("Use the sliders above to adjust input values and instantly see new predictions.")
