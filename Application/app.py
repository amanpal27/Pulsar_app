import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --------------------------------
# Load model & scaler
# --------------------------------
@st.cache_resource
def load_artifacts():
    with open("pulsar_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------
# App UI
# --------------------------------
st.title("🔭 Pulsar Detection App (Random Forest)")
st.write("Detect pulsars from radio signal features")

threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

option = st.sidebar.radio(
    "Input Method",
    ("Upload CSV", "Manual Input")
)

# --------------------------------
# CSV UPLOAD
# --------------------------------
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        X_scaled = scaler.transform(df)
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        df["Pulsar Probability"] = probs
        df["Prediction"] = preds
        df["Label"] = df["Prediction"].map({0: "Noise", 1: "Pulsar"})

        pulsar_count = int(np.sum(preds))

        st.success(f"🟢 Pulsars Detected: {pulsar_count}")
        st.dataframe(df)

        # -----------------------------
        # Confusion Matrix (if labels exist)
        # -----------------------------
        if "target" in df.columns:
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(df["target"], preds)
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Noise", "Pulsar"])
            ax.set_yticklabels(["Noise", "Pulsar"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            st.pyplot(fig)

        # -----------------------------
        # ROC Curve
        # -----------------------------
        st.subheader("ROC Curve")

        if "target" in df.columns:
            fpr, tpr, _ = roc_curve(df["target"], probs)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

# --------------------------------
# MANUAL INPUT
# --------------------------------
else:
    st.subheader("Manual Feature Input")

    feature_names = [
        "Mean_Profile", "Std_Profile", "Kurtosis_Profile", "Skewness_Profile",
        "Mean_DM", "Std_DM", "Kurtosis_DM", "Skewness_DM"
    ]

    inputs = []
    for f in feature_names:
        inputs.append(st.number_input(f, value=0.0))

    if st.button("Detect Pulsar"):
        X = np.array([inputs])
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0, 1]
        pred = int(prob >= threshold)

        if pred == 1:
            st.success(f"🌟 Pulsar Detected (Probability: {prob:.2f})")
        else:
            st.warning(f"❌ Noise (Probability: {prob:.2f})")

# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------
st.markdown("---")
st.subheader("Feature Importance (Random Forest)")

importances = model.feature_importances_
features = feature_names

fi_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.dataframe(fi_df)

fig, ax = plt.subplots()
ax.barh(fi_df["Feature"], fi_df["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
st.pyplot(fig)
# --------------------------------