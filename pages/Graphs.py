import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
import plotly.express as px



# Load your data into a DataFrame
cancer = pd.read_csv('survey_lung_cancer.csv')

import seaborn as sns

# Load your data into a DataFrame
cancer = pd.read_csv('survey_lung_cancer.csv')

# Create a violin plot using Seaborn
fig = sns.violinplot(x='LUNG_CANCER', y='AGE', data=cancer)

# Add labels and title
fig.set(xlabel='Lung Cancer', ylabel='Age', title='Lung Cancer')

# Display the plot in Streamlit
st.pyplot(fig.figure)

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Create a scatter plot using Matplotlib
fig, ax = plt.subplots()

# Plot the points representing people without stroke first
no_stroke = stroke[stroke['stroke'] == 0]
ax.scatter(no_stroke['age'], no_stroke['bmi'], c='lightblue', label='No Stroke', s=10)

# Plot the points representing people with stroke on top
stroke = stroke[stroke['stroke'] == 1]
ax.scatter(stroke['age'], stroke['bmi'], c='red', label='Stroke', s=10)

# Add a legend to show the mapping between color and stroke
ax.legend()

# Add labels and title
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_title('Stroke')

# Display the plot in Streamlit
st.pyplot(fig)


diabetes = pd.read_csv('diabetes_prediction_dataset_kaggle.csv')

fig, ax = plt.subplots()

# Plot the points representing people without stroke first
no_diabetes = diabetes[diabetes['diabetes'] == 0]
ax.scatter(no_diabetes['age'], no_diabetes['bmi'], c='lightblue', label='No Diabetes', s=3)

# Plot the points representing people with stroke on top
diabetes = diabetes[diabetes['diabetes'] == 1]
ax.scatter(diabetes['age'], diabetes['bmi'], c='red', label='Diabetes', s=3)

# Add a legend to show the mapping between color and stroke
ax.legend()

# Add labels and title
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_title('Diabetes')

# Display the plot in Streamlit
st.pyplot(fig)




