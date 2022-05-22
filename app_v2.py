# -*- coding: utf-8 -*-

#####loading libraries ###########################

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#####Introduction to the Webapp and short description###########################

st.set_page_config(
    page_title = "Insurly",
    page_icon = "❤️",
    layout = "wide")
st.title("🩺Insurly's Heart Disease App🏥")
st.markdown("This dashboard helps to explore patient records and analyzing and predicting the risk of a heart disease.")
st.write("The app has been developed by Timon Kayser, Rico Sägesser and Laura Staub")

#####Defining function for loading data and model. Both cached.###########################

@st.cache() #caching makes sure that loaded data and model are stored in caches. If page is reloaded this does not reload again
def load_data():
    data = pd.read_csv("data.csv")
    return(data.dropna())

@st.cache(allow_output_mutation=True)
def load_model():
    filename = "finalized_default_model_v2.sav" 
    loaded_model = pickle.load(open(filename, "rb")) #loading the model that beforehand has been trained
    return(loaded_model)
    loaded_model

data = load_data()
model = load_model()

#####Presenting general information on the dataset###########################
st.header("ℹ️General Information")

row1_col1, row1_col2, row1_col3 = st.columns([1,1,1]) #initialize rows and columns

fig = plt.figure(figsize=(8,3.8))

sns.countplot(data=data, x='DiffWalking', hue='HeartDisease', palette=['#4285f4',"#ea4335"])
plt.title('Diff Walking vs Heart Disease')
plt.xlabel('Diff Walking')
plt.ylabel('Number of Cases')

row1_col1.pyplot(fig, use_container_width=True)

fig2, ax = plt.subplots(figsize = (9,5.5))

ax.hist(data[data["HeartDisease"]==0]["Sex"], bins=3, alpha=0.8, color="#4285f4", label="No HeartDisease")
ax.hist(data[data["HeartDisease"]==1]["Sex"], bins=3, alpha=1, color="#ea4335", label="HeartDisease")

plt.title('Gender Comparison')
ax.set_xlabel("Sex")
ax.set_ylabel("Number of Cases")

ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

row1_col2.pyplot(fig2, use_container_width=True)

fig3 = plt.figure(figsize=(8,3.7))
sns.kdeplot(data[data['HeartDisease']==0]['MentalHealth'],shade=True,color='#4285f4', label='No HeartDisease')
sns.kdeplot(data[data['HeartDisease']==1]['MentalHealth'],shade=True,color='#ea4335', label='Heart Disease')
plt.legend()
plt.title('Variation of Mental Health')

row1_col3.pyplot(fig3, use_container_width=True)

#####Inluding Sliders and Dropdown list to further dive into the data#####################################################
st.header("🔎Patient Explorer")
row2_col1, row2_col2, row2_col3 = st.columns([1,1,1]) #initialize rows and columns

bmi = row2_col1.slider("BMI of the patient",
                min_value=data["BMI"].min(),
                max_value=data["BMI"].max(),
                value=(05.00,20.00)
)

health = row2_col2.slider("General Health of the Patient",
                min_value=float(data["GenHealth"].min()),
                max_value=float(data["GenHealth"].max()),
                value=(1.0,5.0)
)

mask = ~data.columns.isin(["HeartDisease"])
names = data.loc[:, mask].columns
features = row2_col3.selectbox("Select the Variables you would like to compare:", names)

filtered_data = data.loc[(data["BMI"] >= bmi[0]) & # creating filtered data according to the Slider Input of the user
                        (data["BMI"] <= bmi[1]) &
                        (data["GenHealth"] >= health[0]) &
                        (data["GenHealth"] <= health[1]), :]

if st.checkbox("Show filtered data", False):
    st.subheader("Raw Data")
    st.write(filtered_data)
    
#########################################################   
row3_col1, row3_col2 = st.columns([1,1]) #initialize rows and columns

barplotdata = filtered_data[["HeartDisease", features]].groupby("HeartDisease").mean()
fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[features], color = "red")
ax.set_ylabel(features)

row3_col1.subheader("Compare Patient Groups")
row3_col1.pyplot(fig1, use_container_width=True)


fig2 = sns.lmplot(y="GenHealth", x = features, data = filtered_data, order=2,
                  height=4, aspect=1/1, col="HeartDisease", hue="HeartDisease", palette = "Set1")

row3_col2.subheader("General Health Regressions")
row3_col2.pyplot(fig2, use_container_width=True)
###################################################################################################

st.header("Predicting Paitent Heart Disease")
uploaded_data = st.file_uploader("Upload your own data set for predicting heart disease of customers")

if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    row4_col1, row4_col2, row4_col3,row4_col4 = st.columns([1,1,1,1])
    row5_col1, row5_col2, row5_col3,row5_col4 = st.columns([1,1,1,1])
        
    probas = model.predict_proba(new_customers)
    probas_panda  = pd.DataFrame(probas, columns = ['Probability of No Disease','Probability of Disease'])
    
    group1 = probas_panda.loc[(probas_panda['Probability of Disease'] >= 0.95) & (probas_panda['Probability of Disease'] <= 1.00)]
    
    group2 = probas_panda.loc[(probas_panda['Probability of Disease'] >= 0.85) & (probas_panda['Probability of Disease'] < 0.95)]    
       
    group3 = probas_panda.loc[(probas_panda['Probability of Disease'] >= 0.65) & (probas_panda['Probability of Disease'] < 0.85)]
        
    group4 = probas_panda.loc[(probas_panda['Probability of Disease'] >= 0.55) & (probas_panda['Probability of Disease'] < 0.65)]
   

    st.success("You successfully scored new customers")
       
        
    row4_col1.subheader("Risk Group Very High: Markup X")
    row4_col1.write(group1['Probability of Disease'].count())
    final_group1 = group1.sort_values(by=['Probability of Disease'], ascending=False)
    row4_col1.table(final_group1["Probability of Disease"])

    row4_col2.subheader("Risk Group Hig: Markup X")
    row4_col2.write(group2['Probability of Disease'].count())
    final_group2 = group2.sort_values(by=['Probability of Disease'], ascending=False)
    row4_col2.table(final_group2["Probability of Disease"])

    row4_col3.subheader("Risk Group Medium: Markup X")
    row4_col3.write(group3['Probability of Disease'].count())
    final_group3 = group3.sort_values(by=['Probability of Disease'], ascending=False)
    row4_col3.table(final_group3["Probability of Disease"])


    row4_col4.subheader("Risk Group Medium Low: Markup X")
    row4_col4.write(group4['Probability of Disease'].count())
    final_group4 = group4.sort_values(by=['Probability of Disease'], ascending=False)
    row4_col4.table(final_group4["Probability of Disease"])
    
    
    st.download_button(label = "Download Scored Customer Data", 
                           data= new_customers.to_csv().encode("utf-8"),
                           file_name = "scored_customer_data.csv")
#######################################################################




    
    




























