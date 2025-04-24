# 🐧 Palmer Penguins Species Classifier - Streamlit App

This project demonstrates a simple machine learning application built with Streamlit to classify penguin species based on their physical measurements. It uses the popular Palmer Penguins dataset.

## ✨ Features

*   **Data Loading & Overview:** Loads the Palmer Penguins dataset (using Seaborn initially, then processed with Pandas) and displays a sample.
*   **Exploratory Data Analysis (EDA):** Visualizes the distribution of key features (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`) for each penguin species using Streamlit's native bar charts. Also includes a scatter plot showing the relationship between bill length and depth.
*   **Interactive Model Training:**
    *   Allows users to select which features to include in the model via a sidebar multiselect.
    *   Allows users to adjust the train/test split ratio using a slider.
    *   Trains a Logistic Regression model using Scikit-learn upon clicking a button.
    *   Displays the accuracy score on the unseen test set.
*   **Interactive Prediction:**
    *   Provides number input fields for users to enter custom penguin measurements.
    *   Predicts the species for the custom input using the trained model.
*   **Clear Explanations:** Includes expander sections explaining the dataset, the model training process, and the prediction mechanism.

## 📊 Dataset

The app uses the **Palmer Penguins** dataset, collected and made available by [Dr. Kristen Gorman](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) and the [Palmer Station, Antarctica LTER](https://pal.lternet.edu/), a member of the [Long Term Ecological Research Network](https://lternet.edu/).

The dataset contains measurements for three penguin species: Adélie, Gentoo, and Chinstrap.

## 🛠️ Technology Stack

*   **Web Framework:** [Streamlit](https://streamlit.io/)
*   **Data Handling:** [Pandas](https://pandas.pydata.org/)
*   **Machine Learning:** [Scikit-learn](https://scikit-learn.org/stable/)
*   **Data Loading (Initial):** [Seaborn](https://seaborn.pydata.org/) (used primarily for `load_dataset`)

## 🚀 Running the App Locally

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas scikit-learn seaborn
    ```
    *(Note: Seaborn is only needed for `load_dataset` in this version)*

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 🎮 Usage

1.  **Explore Data:** Examine the sample data and the feature distribution charts.
2.  **Configure Model:** Use the sidebar to select features and set the test set size.
3.  **Train Model:** Click the "🚀 Train model" button. The accuracy on the test set will be displayed.
4.  **Predict:** Enter custom measurements in the "🔍 Try it out" section and click "Predict species" to see the model's prediction.

---

*Feel free to modify or add sections as needed!*
