# ── 0. Setup: import the usual suspects ──
import streamlit as st      # 🔤️  Web UI in pure Python
import pandas as pd         # 📊  Data wrangling
import seaborn as sns       # 🐧  Penguin dataset helper
from sklearn.model_selection import train_test_split  # 🔀 Train/test split
from sklearn.preprocessing import StandardScaler      # ⚖️  Feature scaling
from sklearn.linear_model import LogisticRegression   # 🤖  Classifier
from sklearn.metrics import accuracy_score            # 🕏  Metric

# ── 1. Dataset Overview ──
st.markdown("""
# Streamlit Demo app
This Streamlit app was created to showcase some of Streamlit's features. Find out more on the associated blog post and dive into the code on the [repo](https://github.com/cylabstreamlit/streamlit). 

### 🐧 About the Dataset
The **Palmer Penguins** dataset is a popular alternative to the Iris dataset for beginner ML tasks. It records measurements for three penguin species (*Adélie*, *Gentoo*, and *Chinstrap*) from three islands in Antarctica.

**Key features:**
- `bill_length_mm`: Length of the penguin's beak
- `bill_depth_mm`: Depth/thickness of the beak
- `flipper_length_mm`: Flipper length (important for swimming)
- `body_mass_g`: Body weight
- `island`: One of three islands (categorical)
- `species`: Target label (what we want to predict)

We'll use only the numeric measurements to keep it simple.
""")

@st.cache_data
def load_data():
    # 1⃣ Load & drop missing rows for simplicity
    penguins = sns.load_dataset("penguins").dropna()

    # 2⃣ Encode the categorical target so scikit-learn is happy
    species_map = {s: i for i, s in enumerate(penguins["species"].unique())}
    penguins["species_id"] = penguins["species"].map(species_map)
    return penguins, species_map

penguins, species_map = load_data()

st.markdown("### 📟 Sample of the dataset")
st.dataframe(penguins.head())

st.markdown("---")
st.markdown("### 📈 Explore feature distributions")

# Define number of bins for histograms
NUM_BINS = 30

for feature in ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]:
    st.subheader(f"Distribution of {feature} by species")

    # 1. Create bins for the feature
    # We use pd.cut to divide the range of the feature into NUM_BINS intervals
    penguins[f'{feature}_bin'] = pd.cut(penguins[feature], bins=NUM_BINS)

    # 2. Group by bin and species, then count occurrences
    chart_data = penguins.groupby([f'{feature}_bin', 'species']).size().unstack(fill_value=0)

    # 3. Optional: Make bin labels more readable (e.g., "32.1-33.0")
    chart_data.index = chart_data.index.map(lambda interval: f"{interval.left:.1f}-{interval.right:.1f}")

    # 4. Display the data using Streamlit's native bar chart
    st.bar_chart(chart_data)

    # Clean up the temporary bin column (optional)
    penguins = penguins.drop(columns=[f'{feature}_bin']) # Keep if needed later

# ── 2. Sidebar ──
st.sidebar.header("⚙️ Model settings")

features = st.sidebar.multiselect(
    "Choose features",
    [c for c in penguins.columns if penguins[c].dtype != "object" and c != "species_id"],
    default=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
    help="Tick the boxes to include/exclude measurements."
)

test_size = st.sidebar.slider(
    "Test size (fraction of rows)", 0.1, 0.5, 0.2, 0.05,
    help="0.2 means 20% of the data will be unseen during training."
)


# ── 3. Train & evaluate ──
if "train_clicks" not in st.session_state:
    st.session_state.train_clicks = 0

X = penguins[features]
y = penguins["species_id"]

if st.button("🚀 Train model", help="Fit a Logistic Regression on the selected features"):
    st.session_state.train_clicks += 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)

    model = LogisticRegression(max_iter=200)
    model.fit(scaler.transform(X_train), y_train)

    preds = model.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, preds)

    st.success(f"Accuracy on unseen data: {acc:.2%} (run #{st.session_state.train_clicks})")

    st.session_state.model = model
    st.session_state.scaler = scaler

    with st.expander("What did we just do?", expanded=False):
        st.markdown("""
        1. **Split** the data so we don't cheat.
        2. **Scale** features – many models assume comparable numeric ranges.
        3. **Fit** Logistic Regression – a baseline linear classifier.
        4. **Score** on the 20% test set to approximate real-world performance.
        """)


# ── 4. Interactive prediction ──
st.subheader("🔍 Try it out")

col1, col2 = st.columns(2)
with col1:
    bill_length = st.number_input("Bill length (mm)", min_value=30.0, max_value=60.0, value=45.0, step=0.1)
    bill_depth  = st.number_input("Bill depth (mm)",  min_value=13.0, max_value=22.0, value=17.0, step=0.1)
with col2:
    flipper     = st.number_input("Flipper length (mm)", min_value=170.0, max_value=240.0, value=200.0, step=1.0)
    mass        = st.number_input("Body mass (g)",      min_value=2500.0, max_value=6500.0, value=4200.0, step=100.0)

if st.button("Predict species"):
    if "model" not in st.session_state:
        st.warning("Train the model first! 🚦")
    else:
        sample = pd.DataFrame([[bill_length, bill_depth, flipper, mass]], columns=features)
        scaled = st.session_state.scaler.transform(sample)
        pred_id = st.session_state.model.predict(scaled)[0]
        pred_species = {v: k for k, v in species_map.items()}[pred_id]
        st.info(f"🧬 Likely species: **{pred_species}**")

        with st.expander("What does this mean?", expanded=False):
            st.write("""The model projects your inputs into the same scaled space it was
            trained on and outputs the most probable species label. 👈
            Remember: it's only as good as the data it has seen!""")


# ── 5. Quick exploratory chart ──
st.markdown("### 🔍 Feature interactions")
st.markdown("Let's visualize how some features vary across species to understand which ones may be useful for classification.")

st.subheader("📊 Bill dimensions by species")
chart_data = penguins[["bill_length_mm", "bill_depth_mm", "species"]]
st.scatter_chart(chart_data, x="bill_length_mm", y="bill_depth_mm", color="species")
st.caption("Longer bills and shallower depths often indicate Gentoo penguins, for instance.")
