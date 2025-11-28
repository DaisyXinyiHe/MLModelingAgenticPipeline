import os
from agents import Agent, Runner, FileSearchTool, WebSearchTool,ModelSettings
import asyncio
import pandas as pd
import numpy as np
from openai import OpenAI
import json

client = OpenAI()

class feature_analysis_Agent:

  '''
  ## Feature Analysis Agent
  ## --- Conduct exploratory data analysis based on available data.
  ## Decide what variable to use and feature engineering.
  ## --- Available data are in csv format

  '''

  def __init__(self, model:str = 'gpt-4o', user_instructions = '', user_defined_target = ''):
    self.model = model
    self.name = 'Feature Analysis Agent'
    self.user_defined_target = user_defined_target
    self.user_instructions = 'You are a data scientist specialized in feature analysis.\
    You analyze data, gives helpful insights on what variable to use and feature engineering.\
    Then you write code to apply your feature engineering suggesions and transform the dataset. '+user_instructions
    if self.user_defined_target:
      self.user_instructions += f'The target variable is {self.user_defined_target}'
    self.agent = Agent(
        name = self.name,
        model = self.model,
        instructions = self.user_instructions,
        model_settings = ModelSettings(temperature = 0),
        # output_type = feature_analysis_result
        )

  async def run(self, csv_path: str, varb_info_path:str=False):
    df = self._load_csv(csv_path)
    profile = self._profile_data(df)
    suggestions = await self._llm_interpretation(profile, varb_info_path)
    return {
        'raw_profile':profile,
        'Feature_analysis_suggestions':suggestions}

  def _load_csv(self, csv_path:str):
    '''load data'''
    df = pd.read_csv(csv_path)
    return df

  def _profile_data(self, df:pd.DataFrame):
    '''Check dataset profile, such as missing %, dtype, unique counts, etc'''
    profile = {}
    profile['n_rows'] = len(df)
    profile['n_cols'] = len(df.columns)

    ## Check dtype, missing %,
    col_info = df.describe().to_dict()
    for i in df.columns:
      col_info[i]['missing_pct'] = 1 - col_info[i]['count']/len(df)
      col_info[i]['dtype'] = str(df[i].dtype)
      col_info[i]['unique_count'] = len(df[i].unique())

    profile['col_info'] = col_info

    return profile

  async def _llm_interpretation(self, profile_dict, varb_info_path = False):
    '''
    Sends data summary and variable information if any to LLM
    '''
    if varb_info_path:
      with open(varb_info_path, 'r') as file:
        varb_info = file.read()
    else:
      varb_info = 'None'

    prompt = f"""
    Given the following dataset profile:
    {json.dumps(profile_dict, indent = 2)}
    and the variable information below:
    {varb_info}

    Please :
    1. Identify likely target variable if no target provided in instructions; otherwise, use the provided target
    2. Identify useful predictor features
    3. Identify columns to drop and reasoning
    4. Suggest feature engineering (e.g. log transform, bucketization). \
    Any columns identified as 'drop columns' should not be selected for feature engineering.
    5. Summarize missing variable issues and solutions

    Return JSON structured as:
    {{
      'likely_targets':[],
      'selected_features':[],
      'drop_columns':[{{'Feature','Reason'}}],
      'feature_engineering':[{{'Feature','Method','Reason'}}],
      'missing_value_handling':[{{'Feature','Method','Reason'}}]

    }}

    """

    result = await Runner.run(self.agent, prompt)
    print(result.final_output)
    return result.final_output

  async def generate_transformation_code(self, df_profile, suggestions):
    """
    Based on suggestions, write code to transform data.
    """


    prompt = f"""
    Based on the following:
    Data Profile:\n{json.dumps(df_profile, indent = 2)},
    Data Transformation Suggestions:\n{json.dumps(suggestions, indent = 2)},

    Write python code that:
    1. The dataset is called 'df'. Do not change the name of the dataset. Do not \
    read additional data.
    2. Applies suggested transformations from Suggestions
    3. return the transformed data in a pandas dataframe format as 'df_transformed'

    Only return python code. No explanation.
    """


    result = await Runner.run(self.agent, prompt)
    # print(result.final_output)
    return result.final_output

  def execute_code(self, csv_path: str,  code:str):
    '''Executes generated code and return df_transformed'''
    df = pd.read_csv(csv_path)
    local_varbs = {'df':df.copy(), 'pd':pd, 'np': np}
    code = code.replace("```", "")
    code = code.removeprefix("python")
    exec(code, {}, local_varbs)
    df_transformed = local_varbs.get('df_transformed',None)

    return df_transformed
  

class ModelingAgent:

  def __init__(self, model = 'gpt-4o',user_instructions = '', user_defined_target = ''):
    self.model = model
    self.name = 'Modeling Agent'
    self.user_defined_target = user_defined_target
    self.user_instructions = 'You are a data scientist specialized in machine learning modeling.\
    You take in problem statement, variable descriptions, transformed data based on the suggestions by the feature analysis agent, \
     and target variable(s). Based on these information, you make suggestions \
     on: \
     1. What kind of model we should build (e.g. Classification or regression)\
     2. What machine learning algorithm to use (e.g. linear regression,time series, XGBoost, or deep learning)\
     3. What hyperparameter we should tune\
      '+user_instructions

    if self.user_defined_target:
      self.user_instructions += f'The target variable is {self.user_defined_target}'

    self.agent = Agent(
        name = self.name,
        model = self.model,
        instructions = self.user_instructions,
        model_settings = ModelSettings(temperature = 0)
        )


  async def proposed_model(self, problem_statement:str, varb_info_path:str,fea_eng_suggestions:str, df_transformed:pd.DataFrame, target_variable:str):

    if varb_info_path:
      with open(varb_info_path, 'r') as file:
        original_varb_info = file.read()
    else:
      original_varb_info = 'None'


    schema = {
        'columns':list(df_transformed.columns),
        'problem_statement':problem_statement,
        'original_varb_info':original_varb_info,
        'fea_eng_suggestions':fea_eng_suggestions,
        'target_variable':target_variable
    }

    prompt = f"""
    You are given:
    * Problem Statement: {schema.get('problem_statement')}
    * Original Variable Descriptions: {schema.get('original_varb_info')}
    * Feature Engineering Suggestions: {schema.get('fea_eng_suggestions')}
    * Transformed Data: {schema.get('columns')}
    * Target Variable: {schema.get('target_variable')}

    Tasks:
    1. Determine whetehr problem is regression or classification
    2. Select a machine learning model type
    3. Recommend hyperparameters to tune on
    4. Recommend evaluation protocol (cross validation or train/test split only)
    5. If cross validation, recommend train/validation/test split ratio as 0.7:0.2:0.1. \
    If train/test split only, recommend train/test split ratio as 0.8:0.2.
    6. Suggest metrics to evaluate model performance
    7. Explain reasoning of the above decisions
    8. Return JSON strictly:

    {{
      'target_variable':'{target_variable}',
      'features':{str([x for x in list(df_transformed.columns) if x != target_variable])}.
      'task_type':'regression'|'classification',
      'model_type':'linear_regression'|'time_series'|'xgboost'|'deep_learning'|...,
      'hyperparameters':'hyperparameter':'value',
      'evaluation_protocol':'cross_validation'|'train_test_split_only',
      'train_test_split_ratio':0.7:0.2:0.1 | 0.8:0.2,
      'metrics':['metric1','metric2','metrics3'...],
      'reasoning':'reasoning'


    }}

    """

    result = await Runner.run(self.agent, prompt)
    print(result.final_output)
    return result.final_output

  async def generate_modeling_code(self, modeling_proposal:str,):
    """
    Based on modeling_proposal, write code to train model and evaluate model performance.
    """


    prompt = f"""
    Based on the following:
    Modeling_proposal:\n{json.dumps(modeling_proposal, indent = 2)},

    Write python code that:
    1. The dataset is called 'df_transformed'. Do not change the name of the dataset. Do not \
    read additional data.
    2. Applies modeling proposal from Modeling_proposal
    3. Write code to train model and evaluate model performance. \
        a. If you use 'early_stopping_rounds' as a parameter, pass it to the constructor of model.
        b. Save training history
    4. return model as 'Model',  evaluation result as 'evaluation_result':{{'metric1':float,'metric2':float...}}\
    training history as 'training_history'

    Only return python code. No explanation.
    """


    result = await Runner.run(self.agent, prompt)
    print(result.final_output)
    return result.final_output

  async def execute_code(self, df_transformed: pd.DataFrame,  code:str):
    '''Executes generated code and return model and result'''
    local_varbs = {'df_transformed':df_transformed.copy(), 'pd':pd, 'np': np}
    code = code.replace("```", "")
    code = code.removeprefix("python")
    try:
      exec(code, {}, local_varbs)
    except Exception as e:
      prompt = f'''Receive this error: {e}. Fix the error in the original code.Original code: {code}. Only return python code. No explanation.'''
      result = await Runner.run(self.agent, prompt)
      print(f"{prompt} \n Error: {e} \n Updated code: {result.final_output}")
      code = result.final_output
      code = code.replace("```", "")
      code = code.removeprefix("python")
      exec(code, {}, local_varbs)

    model = local_varbs.get('Model',None)
    evaluation_result = local_varbs.get('evaluation_result',None)
    training_history = local_varbs.get('training_history',None)

    return model, evaluation_result,training_history


  