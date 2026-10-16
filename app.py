import streamlit as st
import pandas as pd
from classifier import model, predict_tags, create_output_excel

st.set_page_config(
    page_title="PLC Tag Classifier",
    page_icon="⚙️",
    layout="wide"
)

st.title("PLC Tag Classification System")

st.write(
    "Upload a PLC tag Excel file to classify tags into "
    "Fault, I/O, Maintenance, or Other."
)

uploaded_file = st.file_uploader(
    "Upload PLC Tag Excel",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:

    df_input = pd.read_excel(uploaded_file)

    # Normalize column names
    df_input.columns = (
        df_input.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    required_columns = [
        "plc_tag",
        "access",
        "data_type"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df_input.columns
    ]

    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    else:

        df_input = df_input[
            ["plc_tag", "access", "data_type"]
        ].copy()

        df_input["plc_tag"] = (
            df_input["plc_tag"]
            .astype(str)
            .str.strip()
        )

        df_input["access"] = (
            df_input["access"]
            .astype(str)
            .str.strip()
        )

        df_input["data_type"] = (
            df_input["data_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        st.success(
            f"File loaded successfully — {len(df_input)} tags found."
        )

        st.subheader("Input Tags")

        st.dataframe(
            df_input,
            use_container_width=True
        )

        if st.button("Classify Tags", type="primary"):

            results_df = predict_tags(
                df_input,
                model
            )

            st.session_state["results_df"] = results_df


        if "results_df" in st.session_state:

            results_df = st.session_state["results_df"]

            st.success(
                f"Classification completed — {len(results_df)} tags processed."
            )

            st.subheader("Classification Results")

            st.dataframe(
                results_df,
                use_container_width=True
            )

            if "results_df" in st.session_state:

                results_df = st.session_state["results_df"]

                review_df = results_df[
                    results_df["review_required"] == True
                ]

                if len(review_df) > 0:

                    st.subheader("Manual Review Required")

                    st.warning(
                        f"{len(review_df)} tag(s) require manual classification."
                    )

                    review_labels = {}

                    for idx, row in review_df.iterrows():

                        st.write(f"**PLC Tag:** `{row['plc_tag']}`")

                        selected_category = st.selectbox(
                            "Select category",
                            [
                                "Fault",
                                "I/O",
                                "Maintenance",
                                "Other"
                            ],
                            key=f"review_{idx}"
                        )

                        review_labels[idx] = selected_category

                    if st.button(
                        "Apply Manual Classifications",
                        type="primary"
                    ):

                        label_map = {
                            "Other": 0,
                            "Fault": 1,
                            "I/O": 2,
                            "Maintenance": 3
                        }

                        for idx, category in review_labels.items():

                            results_df.loc[
                                idx,
                                "predicted_label"
                            ] = label_map[category]

                            results_df.loc[
                                idx,
                                "category"
                            ] = category

                            results_df.loc[
                                idx,
                                "source"
                            ] = "Manual Review"

                            results_df.loc[
                                idx,
                                "review_required"
                            ] = False

                            results_df.loc[
                                idx,
                                "review_reason"
                            ] = ""

                        st.session_state["results_df"] = results_df

                        st.success(
                            "Manual classifications applied successfully."
                        )

                        st.rerun()

                if "results_df" in st.session_state:
                
                    results_df = st.session_state["results_df"]

                    pending_count = results_df[
                        results_df["review_required"] == True
                    ].shape[0]

                    if pending_count == 0:

                        st.subheader("Export Results")

                        excel_file = create_output_excel(
                            results_df
                        )

                        st.download_button(
                            label="📥 Download Excel",
                            data=excel_file,
                            file_name="PLC_Tag_Classification.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            )
                        )

                    else:

                        st.info(
                            f"{pending_count} tag(s) still require "
                            "manual classification before export."
                        )
