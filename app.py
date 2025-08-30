import streamlit as st
import pandas as pd
from utils.storage import FeedbackStore, FeedbackItem

st.set_page_config(page_title="AgentSynth Data Inspector", layout="wide")
st.title("AgentSynth – Data Inspector")

data_path = st.sidebar.text_input("Generated CSV path", value="data/generated.csv")
scenario = st.sidebar.text_input("Scenario label", value="evening_rush")
store = FeedbackStore("logs/feedback.sqlite")

@st.cache_data
def load_data(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

df = load_data(data_path)
if df.empty:
    st.warning("No data found. Run `python main.py` to generate.")
else:
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Filter")
    min_price, max_price = float(df["price"].min()), float(df["price"].max())
    price_range = st.slider("Price range", min_value=min_price, max_value=max_price, value=(min_price, max_price))
    cats = st.multiselect("Categories", sorted(df["category"].unique().tolist()), default=df["category"].unique().tolist())
    view = df[(df["price"].between(price_range[0], price_range[1])) & (df["category"].isin(cats))]
    st.write(f"Filtered rows: {len(view)}")
    st.dataframe(view.head(100), use_container_width=True)

    st.subheader("Provide Feedback (Thumbs)")
    row_idx = st.number_input("Row index to label", min_value=0, max_value=max(len(df)-1, 0), step=1)
    col1, col2 = st.columns(2)
    comment = st.text_input("Optional comment", value="")
    with col1:
        if st.button("👍 Mark as Realistic"):
            store.add(FeedbackItem(row_idx=int(row_idx), label=+1, comment=comment, scenario=scenario))
            st.success("Saved 👍 feedback")
    with col2:
        if st.button("👎 Mark as Unrealistic"):
            store.add(FeedbackItem(row_idx=int(row_idx), label=-1, comment=comment, scenario=scenario))
            st.success("Saved 👎 feedback")

    st.subheader("Summary")
    avg = store.get_recent_avg_label()
    st.write(f"Recent average label (±1): {avg if avg is not None else 'n/a'}")
