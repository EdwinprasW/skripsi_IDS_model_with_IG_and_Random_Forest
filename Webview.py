import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the models
with open('ids_multi_model.pkl', 'rb') as file:
    multi_model = pickle.load(file)

with open('ids_binary_model.pkl', 'rb') as file:
    binary_model = pickle.load(file)

# Load the scalers
with open('multi_test_scaler.pkl', 'rb') as file:
    multi_scaler = pickle.load(file)

with open('binary_scaler.pkl', 'rb') as file:
    binary_scaler = pickle.load(file)
    
with open('multi_test_le.pkl', 'rb') as file:
    multi_le = pickle.load(file)

with open('binary_le.pkl', 'rb') as file:
    binary_le = pickle.load(file)


# Function to process the data for multiclass model
def process_data_multi(df):
    df = df.drop(columns='labels')
    # Convert categorical columns to string type
    cols = [ 'service', 'flag']
    for col in cols:
        df[col] = df[col].astype(str)

    # Encoding
    for col in cols:
        le = multi_le[col]
        df[col] = le.transform(df[col])

    top_features = ['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in', 'count',
'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate',
'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
'dst_host_serror_rate', 'dst_host_srv_serror_rate' ]  # Define the top features you want to select

    X_tes_datanew_selected = df[top_features]
    X_tes_datanew_selected = multi_scaler.transform(X_tes_datanew_selected)
    return X_tes_datanew_selected

# Function to process the data for binary model
def process_data_binary(df):
    # Convert categorical columns to string type
    cols = ['protocol_type',    'service', 'flag']
    for col in cols:
        df[col] = df[col].astype(str)

    # Encoding
    for col in cols:
        le = binary_le[col]
        df[col] = le.transform(df[col])


    top_features = ['src_bytes', 'service', 'dst_bytes', 'flag', 'same_srv_rate',
       'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
       'logged_in', 'dst_host_serror_rate', 'dst_host_diff_srv_rate',
       'dst_host_srv_serror_rate', 'serror_rate', 'srv_serror_rate',
       'count', 'dst_host_srv_diff_host_rate', 'dst_host_count',
       'dst_host_same_src_port_rate', 'srv_diff_host_rate', 'srv_count']  # Define the top features you want to select

    X_tes_datanew_selected = df[top_features]
    X_tes_datanew_selected = binary_scaler.transform(X_tes_datanew_selected)
    return X_tes_datanew_selected

# Function to predict using the model
def predict(model, data, process_function):
    processed_data = process_function(data)
    return model.predict(processed_data)

# Function to convert dataframe to CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Define the pages
def page_multiclass():
    st.title("IDS Model Prediction - Multiclass")

    st.header("Upload CSV for Prediction")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if csv_file is not None:
        data = pd.read_csv(csv_file)
        st.write("Data from CSV:")
        st.write(data)

        if st.button("Predict from CSV"):
            predictions = predict(multi_model, data, process_data_multi)

            # Initialize an empty list to hold the results
            results = []

            # Define the labels dictionary
            labels = {
                0: 'DoS',
                1: 'Normal',
                2: 'Probe'
            }

            # Iterate through the array and categorize each value
            for value in predictions:
                results.append(labels.get(value, 'unknown'))  # 'unknown' for any value not in the labels dictionary

            # Add predictions to the dataframe
            data['predictions'] = results

            # Display predictions
            st.write("Predictions:")
            st.write(data)

            # Create a downloadable CSV file
            csv = convert_df(data)

            # Download button to download the result as CSV
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='multiclass_predictions.csv',
                mime='text/csv',
            )

def page_binary():
    st.title("IDS Model Prediction - Binary")

    st.header("Upload CSV for Prediction")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if csv_file is not None:
        data = pd.read_csv(csv_file)
        st.write("Data from CSV:")
        st.write(data)

        if st.button("Predict from CSV"):
            predictions = predict(binary_model, data, process_data_binary)

            # Initialize an empty list to hold the results
            results = []

            # Define the labels dictionary
            labels = {
                0: 'Attack',
                1: 'Normal'
            }

            # Iterate through the array and categorize each value
            for value in predictions:
                results.append(labels.get(value, 'unknown'))  # 'unknown' for any value not in the labels dictionary

            # Add predictions to the dataframe
            data['predictions'] = results

            # Display predictions
            st.write("Predictions:")
            st.write(data)

            # Create a downloadable CSV file
            csv = convert_df(data)

            # Download button to download the result as CSV
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='binary_predictions.csv',
                mime='text/csv',
            )

# Define the main function to handle navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Multiclass Prediction", "Binary Prediction"])

    if page == "Multiclass Prediction":
        page_multiclass()
    else:
        page_binary()

if __name__ == "__main__":
    main()
