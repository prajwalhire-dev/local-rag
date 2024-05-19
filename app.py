#import req lib
from dotenv import load_dotenv, find_dotenv

import os
import streamlit as st
import pandas as pd

from langchain_community.llms import OpenAI 
#from langchain.agents import create_pandas_dataframe_agent 
from langchain_experimental.agents import create_pandas_dataframe_agent


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#main
st.title('AI Assistant for Data Science')
st.write('Hello, I am your assitance and i am her eto help you with your Data science project')

#sidebar
with st.sidebar:
    st.write('*Your data science adventure begins with a csv file. *')
    st.caption("**It's a capstone project where it deals with most of the DS analysis background**")

    st.divider()   


#Initialise the session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

#Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    st.header('Explorartory Data Analysis')
    
    user_csv = st.file_uploader("Upload your file here", type='csv')

    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(temperature=0)

        #Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm('What are the steps of EDA')
            return steps_eda
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose= True)

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        st.header('Exploratory data analysis')
        st.subheader('General information about dataset')

        with st.sidebar:
            with st.expander('what are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study!')
        user_question_variable = st.text_input('What variables are you interested')

        if user_question_variable is not None and user_question_variable != "":
            function_question_variable()

            st.header('Further Study')
        else:
            st.info('Give the variable name')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")


                if user_question_dataframe:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")
                
                    prompt = st.text_input("Add your prompt here")

                    data_problem_template = PromptTemplate(
                        input_variables=['business_problem'],
                        template='Convert the following problem into a data science problem: {business_problem}.'
                    )
                    model_problem_template = PromptTemplate(
                        input_variables=['data_problem'],
                        template='Give a list of Machine learning algorithms that are suitable to solve this problem: {data_problem}.'
                    )

                    data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True, output_key = 'data_problem')
                    model_selection_problem_chain = LLMChain(llm=llm, prompt=model_problem_template, verbose=True, output_key = 'model_selection')

                    sequential_chain = SequentialChain(chains = [data_problem_chain, model_selection_problem_chain], input_variables=['business_problem'],
                                                       output_key=['data_problem','model_selection'],
                                                       verbose=True )
                    if prompt:
                        response = sequential_chain({'business_problem':prompt})
                        st.write(response['data_problem'])
                        st.write(response['model_selection'])



 










