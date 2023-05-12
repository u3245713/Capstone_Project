import streamlit as st
import pandas as pd
from carEval import EDA, PDA

# Set up Streamlit page
st.set_page_config(page_title='Car Evaluation', page_icon=':car:', layout='centered')

# Load the data
data = pd.read_csv('car.data')

# Create instances of the EDA and PDA classes
eda = EDA(data)
pda = PDA(data)


# Streamlit app
def app():
    # Set title of the web page
    st.title('Car Data Analysis')

    # View the first five rows of the data
    st.subheader('View Data')
    st.write(eda.view_data())

    # View the unique values of each attribute
    st.subheader('Attribute Values')
    st.write(eda.attributes_values())

    # Check for missing values
    st.subheader('Missing Values')
    st.write(eda.missing_values())

    # Plot a pie chart of the class distribution
    st.subheader('Pie Chart')
    pie_chart = eda.pie_chart()
    st.pyplot(pie_chart)

    # Plot bar charts for each attribute
    st.subheader('Bar Charts')
    bar_chart = eda.bar_chart()
    st.pyplot(bar_chart)

    # Preprocess the data
    pda.preprocess_data()

    # Compare algorithms and display the results
    st.subheader('Algorithm Comparison')
    st.write(pda.compare_algorithms())
    st.pyplot(pda.display_algorithm_comparison())

    # Display the accuracy of the best model
    st.subheader('Model Accuracy')
    st.write(pda.modelAccuracy())

    # Display the classification report for the best model
    st.subheader('Classification Report')
    report_1, report_2, accuracy = pda.display_classification_report()
    st.write(report_1)
    st.write(report_2)
    st.write(accuracy)


# Run the app
if __name__ == '__main__':
    app()
