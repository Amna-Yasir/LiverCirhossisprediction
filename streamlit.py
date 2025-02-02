import datetime

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

# Title
st.title("Cirrhosis Patient Outcome Prediction")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Introduction", "EDA", "Data Preprocessing", "Modeling"])

# Introduction Section
if page == "Introduction":
    st.header("Introduction")
    st.write("""
    This application predicts the outcomes of patients with cirrhosis based on clinical and medical features.
    The outcomes include:
    1. Censored (patient outcome unknown)
    2. Censored due to liver transplant
    3. Deceased
    """)

# Upload Dataset Section (Only when in "EDA" or "Data Preprocessing")
if page == "EDA" or page == "Data Preprocessing":
    # File uploaders inside EDA/Data Preprocessing sections
    train_data_uploaded_file = st.sidebar.file_uploader("Upload your train dataset (CSV format)", type=["csv"], key="train_data_uploader")
    test_data_uploaded_file = st.sidebar.file_uploader("Upload your test dataset (CSV format)", type=["csv"], key="test_data_uploader")

    # Check if files are uploaded before processing
    if train_data_uploaded_file is not None and test_data_uploaded_file is not None:
        # Load the datasets
        train_data = pd.read_csv(train_data_uploaded_file)
        test_data = pd.read_csv(test_data_uploaded_file)

        # Show the preview of the dataset
        st.subheader("Dataset Preview")
        st.dataframe(train_data.head())

        if page == "EDA":
            # Exploratory Data Analysis (EDA) Section
            st.header("Exploratory Data Analysis (EDA)")
            st.write("### Basic Statistics")
            st.write(train_data.describe())

            st.write("### Data Types")
            st.write(train_data.dtypes)

            st.write("### Unique Value Counts")
            unique_counts = train_data.nunique()
            st.write(unique_counts)

            st.write("### Missing Values")
            st.write(train_data.isnull().sum())

            selected_column = st.selectbox("Select Feature for Distribution Analysis", train_data.columns)

            # Categorical Feature Visualization
            st.header("Categorical Feature Visualization")
            categorical_features = st.multiselect("Select categorical features to visualize", train_data.select_dtypes(include=['object']).columns)

            def plot_categorical(data, features):
                n_col = 2
                n_rows = (len(features) + 1) // n_col
                fig, axes = plt.subplots(n_rows, n_col, figsize=(16, 4 * n_rows))
                axes = axes.flatten()

                for i, feature in enumerate(features):
                    sns.countplot(data=data, x=feature, ax=axes[i], palette='Paired')
                    for p in axes[i].patches:
                        axes[i].annotate(f'{p.get_height():.0f}',
                                         (p.get_x() + p.get_width() / 2, p.get_height()),
                                         ha='center', va='center', size=12, xytext=(0, 5), textcoords='offset points')

                    axes[i].set_title(feature)
                    axes[i].set_ylabel('Count')

                if len(features) % 2 != 0:
                    fig.delaxes(axes[-1])

                plt.tight_layout()
                st.pyplot(plt)

            if categorical_features:
                plot_categorical(train_data, categorical_features)

        elif page == "Data Preprocessing":
            # Data Preprocessing Steps
            st.header("Data Preprocessing")

            # Display Preprocessing Steps
            st.subheader("Preprocessing Steps")
            st.write("""
            1. **Replaced 'id' with 'N' in the 'Hepatomegaly' column**:
               - This ensures consistency by converting incorrect values to the 'N' category.

            2. **Handled missing values for numerical columns using IterativeImputer**:
               - Imputed missing values in numerical columns such as 'Age', 'Bilirubin', etc., using Iterative Imputation.

            3. **Handled missing values for categorical columns using proportional imputation**:
               - Categorical columns like 'Drug', 'Sex', 'Ascites', etc., were imputed based on the distribution of existing values.

            4. **One-hot encoded nominal categorical columns**:
               - One-hot encoding was applied to categorical columns like 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', etc.

            5. **Scaled numerical columns using RobustScaler**:
               - Numerical columns like 'Age', 'Bilirubin', etc., were scaled using RobustScaler to handle outliers effectively.
            """)

            # Proceed with Preprocessing
            # Replacing 'id' with 'N' in the 'Hepatomegaly' column
            train_data['Hepatomegaly'] = train_data['Hepatomegaly'].replace('id', 'N')
            test_data['Hepatomegaly'] = test_data['Hepatomegaly'].replace('id', 'N')

            # Handling missing values for numerical columns using IterativeImputer
            num_features = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
            
            iter_imputer = IterativeImputer(max_iter=20, random_state=0)
            
            # Apply the imputation
            train_data[num_features] = iter_imputer.fit_transform(train_data[num_features])
            test_data[num_features] = iter_imputer.transform(test_data[num_features])

            # Handle missing values for categorical columns using proportional imputation
            cat_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
            
            # Impute categorical columns
            for col in cat_cols:
                prop = train_data[col].value_counts(normalize=True)
                fill_values = []
                for value, count in prop.items():
                    fill_values.extend([value] * int(count * train_data[col].isnull().sum()))
                np.random.shuffle(fill_values)

                if len(fill_values) < train_data[col].isnull().sum():
                    fill_values.extend([prop.idxmax()] * (train_data[col].isnull().sum() - len(fill_values)))
                elif len(fill_values) > train_data[col].isnull().sum():
                    fill_values = fill_values[:train_data[col].isnull().sum()]
                
                train_data.loc[train_data[col].isnull(), col] = fill_values

            for col in cat_cols:
                prop = test_data[col].value_counts(normalize=True)
                fill_values = []
                for value, count in prop.items():
                    fill_values.extend([value] * int(count * test_data[col].isnull().sum()))
                np.random.shuffle(fill_values)

                if len(fill_values) < test_data[col].isnull().sum():
                    fill_values.extend([prop.idxmax()] * (test_data[col].isnull().sum() - len(fill_values)))
                elif len(fill_values) > test_data[col].isnull().sum():
                    fill_values = fill_values[:test_data[col].isnull().sum()]
                
                test_data.loc[test_data[col].isnull(), col] = fill_values

            # One-hot encode categorical columns (nominal)
            cat_cols_nominal = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
            nominal_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            train_nominal_encoded = nominal_encoder.fit_transform(train_data[cat_cols_nominal])
            test_nominal_encoded = nominal_encoder.transform(test_data[cat_cols_nominal])

            # feature engg
            # convert num cols to cat

            # Import necessary libraries for data preprocessing
            from sklearn.impute import SimpleImputer, IterativeImputer
            from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
            from imblearn.over_sampling import SMOTE

            #The target variable (the column you're trying to predict) is named 'Status'. This is usually the dependent variable in a machine learning model.
            target= 'Status'

            #(values that don't have an inherent order). 
            cat_cols_nominal = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
            # values that have a meaningful order, 
            cat_cols_ordinal = ['Stage']

            #Numeerical Coloumn
            num_features = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

            # Convert categorical columns to string type
            for col in cat_cols_nominal + cat_cols_ordinal:
                train_data[col] = train_data[col].astype(str)
                test_data[col] = test_data[col].astype(str)


            
            # Scale numerical columns using RobustScaler
            scaler = RobustScaler()
            train_data[num_features] = scaler.fit_transform(train_data[num_features])
            test_data[num_features] = scaler.transform(test_data[num_features])

            # Encode nominal categorical columns using OneHotEncoder
            nominal_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            train_nominal_encoded = nominal_encoder.fit_transform(train_data[cat_cols_nominal])
            test_nominal_encoded = nominal_encoder.transform(test_data[cat_cols_nominal])

            #performing encoding on categorical column
#performing encoding on categorical column
            ordinal_encoder = OrdinalEncoder()

            train_ordinal_encoder = ordinal_encoder.fit_transform(train_data[cat_cols_ordinal])
            test_ordinal_encoder = ordinal_encoder.fit_transform(test_data[cat_cols_ordinal])

            # Create DataFrame for encoded ordinal features
            train_ordinal_df = pd.DataFrame(train_ordinal_encoder, columns=cat_cols_ordinal)
            test_ordinal_df = pd.DataFrame(test_ordinal_encoder, columns=cat_cols_ordinal)

            # Create DataFrame for encoded nominal features
            train_nominal_df = pd.DataFrame(train_nominal_encoded, columns=nominal_encoder.get_feature_names_out(cat_cols_nominal))
            test_nominal_df = pd.DataFrame(test_nominal_encoded, columns=nominal_encoder.get_feature_names_out(cat_cols_nominal))
              
            # Reset indices
            train_data.reset_index(drop=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            # Concatenate processed numerical and categorical features
            train_processed = pd.concat([train_data[num_features], train_nominal_df, train_ordinal_df], axis=1)
            test_processed = pd.concat([test_data[num_features], test_nominal_df, test_ordinal_df], axis=1)
            # Encode target variable
            target_encoder = LabelEncoder()
            y_train = target_encoder.fit_transform(train_data[target])
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(train_processed, y_train)

            # Encode target variable
            target_encoder = LabelEncoder()
            y_train = target_encoder.fit_transform(train_data[target])
            # Display updated datasets after preprocessing
            st.subheader("Updated Training Dataset Preview")
            st.dataframe(train_data.head())

            st.subheader("Updated Test Dataset Preview")
            st.dataframe(test_data.head())

            # Display completion message
            st.write("### Data preprocessing is complete!")

            from sklearn.model_selection import train_test_split
            X_train,X_test,y_train,y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                # accuracy 93
            }

            # Random Forest with GridSearchCV
            rf_clf = RandomForestClassifier()
            rf_grid_search = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, cv=5, scoring='neg_log_loss', n_jobs=-1)
            rf_grid_search.fit(X_train, y_train)

            # Best Random Forest model
            rf_best_model = rf_grid_search.best_estimator_
            rf_best_params = rf_grid_search.best_params_
            rf_best_logloss = -rf_grid_search.best_score_

            # Evaluate Random Forest on test data
            rf_y_pred = rf_best_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_y_pred)


        

           # Display results in Streamlit
            st.subheader("Random Forest Results")

            # Using st.markdown to style the output
            st.markdown(f"**Best Parameters:** {rf_best_params}")
            st.markdown(f"**Best Log Loss:** {rf_best_logloss:.4f}")
            st.markdown(f"**Accuracy:** {rf_accuracy:.4f}")

            # Optionally add some styling or images
            st.markdown("### Conclusion:")
            st.write("This model demonstrates good performance with the selected parameters and accuracy. Further tuning could improve results.")
                        