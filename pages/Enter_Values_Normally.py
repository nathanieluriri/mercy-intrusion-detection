import streamlit as st
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import streamlit as st





def le(df):
    """
    Apply label encoding to object-type columns in the DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame.

    Returns:
        None
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])






def preprocess_data(train, test):
    """
    Preprocess the train and test data for machine learning.

    Parameters:
        train (DataFrame): Train dataset.
        test (DataFrame): Test dataset.

    Returns:
        X_train (array-like): Processed features for training.
        X_test (array-like): Processed features for testing.
        Y_train (array-like): Target variable for training.
        label_encoders (dict): Fitted label encoders.
    """
    le(train)
    le(test)

    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    # Feature selection
    X_train = train.drop(['class'], axis=1)
    Y_train = train['class']
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=10)
    rfe = rfe.fit(X_train, Y_train)

    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
    selected_features = [v for i, v in feature_map if i==True]

    X_train = X_train[selected_features]
    X_test = test[selected_features]

    # Scaling
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    return X_train, X_test, Y_train, le








def process_data(data):
    """
    Preprocess the data for inference.

    Parameters:
        data (DataFrame): Input data.

    Returns:
        X (array-like): Processed features.
    """
    le(data)

    selected_features=['protocol_type', 'service',  'flag',  'src_bytes',  'dst_bytes',  'count',  'same_srv_rate',  'diff_srv_rate',  'dst_host_srv_count',  'dst_host_same_srv_rate']
    X = data[selected_features]
    scale = StandardScaler()
    X = scale.fit_transform(X)

    return X









from sklearn.preprocessing import LabelEncoder
import pandas as pd

def fit_label_encoder(df):
    """
    Fit a label encoder on each column with object dtype in the DataFrame.
    
    Parameters:
        df (DataFrame): Input DataFrame
        
    Returns:
        label_encoder (dict): Dictionary containing fitted label encoders for each column
    """
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(df[col])
            label_encoders[col] = label_encoder
    return label_encoders



def transform_single_row(single_row, label_encoders):
    """
    Transform a single row of data using the fitted label encoders.
    
    Parameters:
        single_row (dict): Dictionary containing data for a single row
        label_encoders (dict): Dictionary containing fitted label encoders for each column
        
    Returns:
        encoded_single_row (dict): Dictionary containing encoded values for the single row
    """
    encoded_single_row = {}
    for col, value in single_row.items():
        if col in label_encoders:
            encoded_value = label_encoders[col].transform([value])[0]
            encoded_single_row[col] = encoded_value
        else:
            # If the value is not found in the fitted label encoder, handle it as needed
            encoded_single_row[col] = value
    return encoded_single_row
st.set_page_config(layout="centered",initial_sidebar_state="expanded",page_title="Network Intrusion Detection",page_icon="🧠")
if "result" not in st.session_state:
    st.session_state.result=None

def combine_dataframe(df0,df1):
    # Get the prediction dataframe from the session state
    df1 = pd.DataFrame(df1, columns=["Predicted results"])
    df0= pd.DataFrame(df0,columns=[
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", 
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
    "dst_host_srv_rerror_rate"
]
)
    df1['Predicted results'] = df1['Predicted results'].astype(object)

    df1.loc[df1['Predicted results'] == 0, "Predicted results"] = 'normal'
    df1.loc[df1['Predicted results'] == 1, "Predicted results"] = 'Attack on network detected'



    # Add the original prediction dataframe as a new column
    combined_df = pd.concat([df0,df1],axis=1)
    

    return combined_df


def create_result_dataframe(prediction_df,df1):
    # Get the prediction dataframe from the session state
    
    
    # Perform model fitting and prediction
    result_from_users = st.session_state.model.predict(prediction_df)
    if result_from_users == 0:
        result_from_users = ["normal"]
    else:
        result_from_users= ["Attack on network detected"]

    # Create a new DataFrame with the prediction results
    result_df = pd.DataFrame({'Prediction Results': result_from_users},index=range(1))

    # Add the original prediction dataframe as a new column
    combined_df = pd.concat([df1,result_df], axis=1)

    return combined_df




if "df" not in st.session_state:
    st.session_state.df=None


with open("models/KNN_model.pkl", "rb") as a:
    KNN_model= pickle.load(a)

with open("models/DecisionTreeClassifier.pkl","rb") as m:
    DTC_model= pickle.load(m)

if "model" not in st.session_state:
    st.session_state.model=KNN_model


def change_model():
    
    if st.session_state.model_select ==  "KNN Model (K-Nearest Neighbors)":
        st.session_state.model= KNN_model
    elif st.session_state.model_select == "Decision Tree classifier":
        st.session_state.model= DTC_model
    


st.title("Network Intrusion Detection")
selected_features = [  'src_bytes',  'dst_bytes',  'count',  'same_srv_rate',  'diff_srv_rate',  'dst_host_srv_count',  'dst_host_same_srv_rate']

for opt in selected_features:
    st.number_input(f" Enter {opt}",key=opt, min_value=0)

st.selectbox("Enter Protocol type",options=["tcp","udp","icmp"],index=None,key="protocol_type")

st.selectbox("Enter Protocol type",options=["SF","S0","REJ","OTH","RSTR"],index=None,key="flag")

st.selectbox("Enter Value for service",options=["ftp_data","other","private","http","remote_job","name","netbios_ns","eco_i","mtp","telnet","finger","domain_u","supdup","uucp_path","Z39_50","smtp","auth","netbios_dgm","csnet_ns","bgp","ecr_i","gopher","vmnet","systat","http_443","efs","imap4","whois","iso_tsap"],index=None,key="service")

if st.session_state.protocol_type and st.session_state.flag and st.session_state.service:        
    
    data = {'protocol_type': [st.session_state.protocol_type],'service': [st.session_state.service], 'flag': [st.session_state.flag], 'src_bytes':[st.session_state.src_bytes], 'dst_bytes': [st.session_state.dst_bytes], 'count': [st.session_state.count], 
    'same_srv_rate':[st.session_state.same_srv_rate], 'diff_srv_rate': [st.session_state.diff_srv_rate], 'dst_host_srv_count': [st.session_state.dst_host_srv_count], 'dst_host_same_srv_rate': [st.session_state.dst_host_same_srv_rate]}
    st.session_state.df=pd.DataFrame(data,index=range(1))
    st.session_state.pf= pd.DataFrame(data,index=range(1))

    st.write(st.session_state.df)
def predict():
    le(st.session_state.df)
    st.session_state.result = create_result_dataframe(st.session_state.df,st.session_state.pf)
    st.write(st.session_state.result)

st.radio(label="Select a model",options=["KNN Model (K-Nearest Neighbors)","Decision Tree classifier"],on_change=change_model,key="model_select")
if st.button("make a prediction",type="primary"):
    predict()