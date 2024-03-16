## BMI
# Import Libraries
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import scipy


# Get the data
train_dataset = pd.read_csv("bmi_data/bmi_train.csv")
test_dataset = pd.read_csv("bmi_data/bmi_validation.csv")

# Set page config
st.set_page_config(
    page_title="Body Mass Index",
    page_icon="🚹"
)


st.title("Body Mass Index")
st.write('''
    World Health Organization (WHO) define overweight and obesity as a condition of abnormal or excessive fat accumulation 
    that presents a risk to health. Overweight and obesity is a medical problem that can increase several health problems 
    or other disaseas. Health problems caused by too much body fat include heart disease, diabetes, high blood pressure, 
    high cholesterol, liver disease, sleep apnea and certain cancers.
         
    A Body Mass Index (also known as BMI) is often used to diagnose obesity. To calculate BMI, divide weight in kilograms 
    by height in meters squared. A BMI over 25 is considered overweight and over 30 is obese.
         
    This section will show an example data of Index with its height and weight data details. Then how BMI is calculated
    with data prepared. At the end, a rule of the BMI mapping will be shown to illustrate how the collaboration between
    height and weight affect the Index.
    ''')


st.header("Dataset")
st.write('''
        The data used in this analysis is shown as below.
        We show only the first 10 rows of data. The full dataset can be obtained from the source link.
    ''')
st.link_button("Source Link", "https://www.kaggle.com/datasets/sjagkoo7/bmi-body-mass-index/data")
st.table(train_dataset.head(10))
st.write('''
    This data contains four columns
    \n- Gender : The gender of the individual
    \n- Height : The height of the individual in cm
    \n- Weight : The weight of individual in kg
    \n- Index : The BMI index of the individual , categorized as follows:
    - 0 : Extremly Weak
    - 1 : Weak
    - 2 : Normal
    - 3 : Overweight
    - 4 : Obesity
    - 5 : Extremly Obesity
    ''')

st.header("BMI Calculation")
st.write('''
        BMI can be calculated by knowing someone's height and weight. 
        Calculating someone's BMI can be performed by using this formula :
    ''')
st.latex(r'''
    BMI = \frac{Weight}{(Height)^2}
    ''')
st.write('''
        where weight is measured in kilograms (kg) and height in meters (m).
        By using this formula, BMI can be calculated as data shown below.
    ''')

train_dataset["BMI"] = train_dataset["Weight"] / ((train_dataset["Height"]/100)*(train_dataset["Height"]/100))
st.table(train_dataset.head(10))

st.header("BMI and Index Analysis")
st.write('''
        Let's see how the BMI and the Index are correlated. First, the Index will be plotted along the BMI calculated.
    ''')

## Chart 1 (scatter plot BMI vs Index)
scatter1 = alt.Chart(train_dataset).mark_point(size=20).encode(
    x=alt.X("BMI:Q", title="Body Mass Index").scale(zero=True),
    y=alt.Y("Index:Q", title="Index").scale(zero=True),
    color=alt.Color("Gender", scale=alt.Scale(domain=['Male', 'Female'] , range=['#33F6FF', '#FF00C5'] )) 
    )
fig = (scatter1).configure_axis(
        labelFontSize=15
    ).properties(
        title=('BMI vs Index'),
        width=650,
        height=400
    ).configure_legend(
        strokeColor='gray',
        padding=10,
        cornerRadius=10,
        orient='bottom-right'
    )
st.altair_chart(fig)

spearman_bmi_index = train_dataset['BMI'].corr(train_dataset['Index'], method='spearman')

st.write('''
        The data looks like it is plotted in an organized manner. Let's try to calculate the correlation
        between BMI and Index. 
         
        To choose what correlation calculation method will be used, each kind of data have to be examined.
        Ignore the Gender variable, the BMI is numerical continuous data and the Index is numerical ordinal 
        data. So one method that is suitable to calculate the correlation is Spearman's Rank Correlation. 
         
        With this method, the correlation calculated is 
    ''', 
    round(spearman_bmi_index,4),
    '''
        . This data is categorized have a high positive correlation. It means the high BMI will have high Index, and
        vice versa. But this number only show that there is a high correlation between BMI and Index. This number doesn't
        explain what variable cause the other variable.

        Let's look again at plotted data above. If some pair of data are excluded from the dataset, it will show
        a clear separation of group of BMI along with the Index. Exclude data with  red mark and then
        the separation group will follow this rule :
        - Index 0 (Extremely Weak) \t: BMI <= 15
        - Index 1 (Weak) : 15 < BMI <= 20
        - Index 2 (Normal) : 20 < BMI <= 25
        - Index 3 (Overweight) : 25 < BMI <= 30
        - Index 4 (Obesity) : 30 < BMI <= 40
        - Index 5 (Extremely Obesity) : BMI > 40
        
    ''')

conditions = [
        (train_dataset['Index'] == 0) & (train_dataset['BMI'] <= 15),
        (train_dataset['Index'] == 1) & (train_dataset['BMI'] > 15) & (train_dataset['BMI'] <= 20),
        (train_dataset['Index'] == 2) & (train_dataset['BMI'] > 20) & (train_dataset['BMI'] <= 25),
        (train_dataset['Index'] == 3) & (train_dataset['BMI'] > 25) & (train_dataset['BMI'] <= 30),
        (train_dataset['Index'] == 4) & (train_dataset['BMI'] > 30) & (train_dataset['BMI'] <= 40),
        (train_dataset['Index'] == 5) & (train_dataset['BMI'] > 40)
    ]
values = ['Included', 'Included', 'Included', 'Included', 'Included', 'Included']
train_dataset["Flag"] = np.select(conditions, values)
train_dataset["Flag"] = np.where(train_dataset["Flag"] == "0", "Excluded", train_dataset["Flag"])
# st.table(train_dataset)   # Check data

## Chart 2 (scatter plot BMI vs Index with mark)
scatter1 = alt.Chart(train_dataset).mark_point(size=20).encode(
    x=alt.X("BMI:Q", title="Body Mass Index").scale(zero=True),
    y=alt.Y("Index:Q", title="Index").scale(zero=True),
    color=alt.Color("Flag", scale=alt.Scale(domain=['Included', 'Excluded'] , range=['#00C391', '#FF5858'] )) 
    )
fig = (scatter1).configure_axis(
        labelFontSize=15
    ).properties(
        title=('BMI vs Index (with marked-excluded data)'),
        width=650,
        height=400
    ).configure_legend(
        strokeColor='gray',
        padding=10,
        cornerRadius=10,
        orient='bottom-right'
    )
st.altair_chart(fig)


st.header("Index Mapping")
st.write('''
        By using the defined rule for indexing the BMI, the Index can be mapped along with various weight and height.
    ''')

map_table = pd.Series(range(140,200))
map_table = pd.DataFrame(map_table)
map_table = map_table.rename(columns={0: "Height"})

weight = pd.Series(range(50,160))
weight = pd.DataFrame(weight)

map_table = map_table.merge(weight, how='cross')
map_table = map_table.rename(columns={0: "Weight"})
map_table["BMI"] = map_table["Weight"] / ((map_table["Height"]/100)*(map_table["Height"]/100))

map_conditions = [
        (map_table['BMI'] <= 15),
        (map_table['BMI'] > 15) & (map_table['BMI'] <= 20),
        (map_table['BMI'] > 20) & (map_table['BMI'] <= 25),
        (map_table['BMI'] > 25) & (map_table['BMI'] <= 30),
        (map_table['BMI'] > 30) & (map_table['BMI'] <= 40),
        (map_table['BMI'] > 40)
    ]
map_values = [0, 1, 2, 3, 4, 5]
map_table["Index"] = np.select(map_conditions, map_values)

## Chart 3 (scatter plot mapping index)
scatter1 = alt.Chart(map_table).mark_point(size=40, filled=True).encode(
    x=alt.X("Weight:Q", title="Weight (in kg)").scale(zero=False),
    y=alt.Y("Height:Q", title="Height (in cm)").scale(zero=False),
    color=alt.Color("Index", scale=alt.Scale(domain=[0,1,2,3,4,5] , range=['#50BAC6','#88D5E1','#A6BBC4','#C3A0A6','#E18689','#FF6B6B'] )) 
    )
fig = (scatter1).configure_axis(
        labelFontSize=15
    ).properties(
        title=('BMI Category'),
        width=650,
        height=400
    )
st.altair_chart(fig)


st.header("Closing")
st.write('''
        Closing statement.
    ''')