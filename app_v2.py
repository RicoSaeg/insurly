# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title = "🏥Heart Disease App🏥",
    page_icon = "❤️",
    layout = "wide")

@st.cache()
def load_data():
    data = pd.read_csv("data.csv")
    return(data.dropna())

@st.cache(allow_output_mutation=True)
def load_model():
    filename = "finalized_default_model_v2.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model) 

data = load_data()
model = load_model()
################################

st.title("Insurly's Heart Disease App")
st.markdown("This dashboard helps to explore patient records and analyzing and predicting the risk of a heart disease.")
st.write("The app has been developed by Timon Kayser, Rico Sägesser and Laura Staub")
###################################

st.header("Patient Explorer")

row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

bmi = row1_col1.slider("BMI of the patient",
                min_value=data["BMI"].min(),
                max_value=data["BMI"].max(),
                value=(05.00,20.00)
)
########################################################


health = row1_col2.slider("General Health of the Patient",
                min_value=float(data["GenHealth"].min()),
                max_value=float(data["GenHealth"].max()),
                value=(1,5)
)
##########################################################
mask = ~data.columns.isin(["HeartDisease", "BMI", "employment_status"])
names = data.loc[:, mask].columns
features = row1_col3.selectbox("Select the Variables you would like to compare:", names)
#########################################################

filtered_data = data.loc[(data["BMI"] >= bmi[0]) &
                        (data["BMI"] <= bmi[1]) &
                        (data["GenHealth"] >= health[0]) &
                        (data["GenHealth"] <= health[1]), :]
################################################

if st.checkbox("Show filtered data", False):
    st.subheader("Raw Data")
    st.write(filtered_data)
#########################################################   

row2_col1, row2_col2 = st.columns([1,1])
########################################################


barplotdata = filtered_data[["HeartDisease", features]].groupby("HeartDisease").mean()
fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[features], color = "green")
ax.set_ylabel(features)

row2_col1.subheader("Compare Patient Groups")
row2_col1.pyplot(fig1, use_container_width=True)
## seaborn plot #########################

fig2 = sns.lmplot(y="BMI", x = features, data = filtered_data, order=2,
                  height=4, aspect=1/1, col="HeartDisease", hue="HeartDisease", palette = "Set2")


row2_col2.subheader("BMI Correlations")
row2_col2.pyplot(fig2, use_container_width=True)

#######################################################################
st.header("Predicting Paitent Heart Disease")
uploaded_data = st.file_uploader("Upload your own data set for predicting heart disease of customers")

if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    
    new_customers["predicted_heart_disease"] = model.predict(new_customers)
    st.success("You successfully scored new customers")
    st.write(new_customers)
    
    
    st.download_button(label = "Download Scored Customer Data", 
                           data= new_customers.to_csv().encode("utf-8"),
                           file_name = "scored_customer_data.csv")


#######################################################################


# age = row1_col3.multiselect(
#    'What is your age category',
#    data["AgeCategory"]
#    )

# st.write('You selected:', age)

######################################################




#st.write("""**2. Select Gender :**""")
#sex = st.selectbox(data["Sex"], [0,1])
#st.write("""**You selected this option **""",sex)

#variable = row1_col2.selectbox("select", names)

#row1_col2.write(variable)



#age = row1_col2.slider("Age Group of the Patients",
              #   min_value=data["AgeCategory"],
               # max_value=data["AgeCategory"],
               #  value=(5-10, 35-45)
                # )






#######################################################################

    
    




























