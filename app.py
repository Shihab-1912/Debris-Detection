from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.model_utils import (
    CATEGORICAL_INPUTS,
    DATA_PATH,
    LABEL_MAP,
    NUMERIC_INPUTS,
    build_prediction_frame,
    clean_data,
    load_model,
    load_raw_data,
)


st.set_page_config(
    page_title="Space Debris Risk Detection",
    page_icon=":satellite:",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return clean_data(load_raw_data(DATA_PATH))


@st.cache_resource
def get_model() -> dict:
    return load_model()


artifact = get_model()
df = get_data()
pipeline = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
metrics = artifact["metrics"]

st.title("Space Debris Risk Detection")
st.caption("Random Forest model trained from the notebook workflow and packaged for Streamlit deployment.")

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
metric_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
metric_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
metric_cols[3].metric("F1 Score", f"{metrics['f1']:.2%}")

tab_predict, tab_explore, tab_data = st.tabs(["Predict", "Explore", "Dataset"])

with tab_predict:
    st.subheader("Predict debris risk level")

    left, right = st.columns([1, 1])
    values: dict[str, object] = {}

    with left:
        st.markdown("#### Orbital measurements")
        for column in NUMERIC_INPUTS[:4]:
            series = pd.to_numeric(df[column], errors="coerce")
            values[column] = st.number_input(
                column.replace("_", " "),
                value=float(series.median()),
                min_value=float(series.min()),
                max_value=float(series.max()),
            )

    with right:
        st.markdown("#### Object measurements")
        for column in NUMERIC_INPUTS[4:]:
            series = pd.to_numeric(df[column], errors="coerce")
            values[column] = st.number_input(
                column.replace("_", " "),
                value=float(series.median()),
                min_value=float(series.min()),
                max_value=float(series.max()),
                format="%.6f",
            )

    cat_cols = st.columns(3)
    for index, column in enumerate(CATEGORICAL_INPUTS):
        options = sorted(df[column].dropna().astype(str).unique().tolist())
        default_value = options[0] if options else "unknown"
        values[column] = cat_cols[index % 3].selectbox(
            column.replace("_", " "),
            options=options or ["unknown"],
            index=(options.index(default_value) if default_value in options else 0),
        )

    if st.button("Predict risk", type="primary"):
        prediction_frame = build_prediction_frame(values, feature_columns)
        prediction = int(pipeline.predict(prediction_frame)[0])
        probabilities = pipeline.predict_proba(prediction_frame)[0]

        st.success(f"Predicted Risk Level: {LABEL_MAP[prediction]}")

        probability_df = pd.DataFrame(
            {
                "Risk Level": [LABEL_MAP[int(class_id)] for class_id in pipeline.classes_],
                "Probability": probabilities,
            }
        )
        st.plotly_chart(
            px.bar(
                probability_df,
                x="Risk Level",
                y="Probability",
                color="Risk Level",
                range_y=[0, 1],
                text_auto=".1%",
            ),
            use_container_width=True,
        )

with tab_explore:
    st.subheader("Dataset overview")
    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.plotly_chart(
            px.histogram(
                df,
                x="Altitude_km",
                nbins=40,
                title="Orbital altitude distribution",
            ),
            use_container_width=True,
        )

    with chart_cols[1]:
        st.plotly_chart(
            px.scatter(
                df,
                x="Altitude_km",
                y="Size_m",
                color="Risk_Level",
                hover_data=["Debris_Type", "Country", "Status"],
                title="Altitude vs debris size",
            ),
            use_container_width=True,
        )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        risk_counts = df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        st.plotly_chart(
            px.bar(
                risk_counts,
                x="Risk Level",
                y="Count",
                color="Risk Level",
                title="Original risk label distribution",
            ),
            use_container_width=True,
        )

    with chart_cols[1]:
        debris_counts = df["Debris_Type"].value_counts().head(10).reset_index()
        debris_counts.columns = ["Debris Type", "Count"]
        st.plotly_chart(
            px.bar(
                debris_counts,
                x="Count",
                y="Debris Type",
                orientation="h",
                title="Top debris types",
            ),
            use_container_width=True,
        )

with tab_data:
    st.subheader("Cleaned dataset sample")
    st.dataframe(df.head(500), use_container_width=True)

    st.download_button(
        "Download cleaned sample",
        df.to_csv(index=False).encode("utf-8"),
        file_name="space_debris_cleaned.csv",
        mime="text/csv",
    )
