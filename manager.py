from ml_agents import feature_analysis_Agent,ModelingAgent,EvaluationAgent,ReportAgent
import asyncio
from openai import OpenAI
from agents import Agent, Runner, ModelSettings

import json
import traceback

client = OpenAI()

class ManagerAgent():

    def __init__(self):
        self.name = 'Manager_Agent'
        self.model = "gpt-4.1"
        self.instruction = "You are the AI Manager Agent orchestrating a pipeline of specialized agents.\
                Your job is to manage workflow, route tasks to each agent in sequence, check for errors,\
                and compile outputs. You do not do analysis directly."
        # self.user_instructions= user_instructions
        # self.user_defined_target = user_defined_target
        # self.csv_path = csv_path
        # self.varb_info_path = varb_info_path
        self.fea_agent = feature_analysis_Agent()
        self.modeling_agent = ModelingAgent()
        self.evaluation_agent = EvaluationAgent()
        self.report_agent = ReportAgent()
        
        # Register manager agent
        self.agent = Agent(
            name = self.name,
            model = self.model,
            instructions = self.instruction,
            model_settings = ModelSettings(temperature = 0)
        )

    async def run_pipeline(self,user_instructions,user_defined_target,csv_path,varb_info_path):
        '''
        user_instructions: problem statement
        user_defined_target: Predict target
        csv_path: train data path
        varb_info_path: Variable information path
        '''

        ## Feature analysis agent
        self.fea_agent.user_instructions = user_instructions
        self.fea_agent.user_defined_target=user_defined_target
        fea_eng_result = await self.fea_agent.run(csv_path = self.csv_path, varb_info_path=self.varb_info_path)

        code = await self.fea_agent.generate_transformation_code(
            fea_eng_result['raw_profile'],
            fea_eng_result['Feature_analysis_suggestions']
        )
        df_transformed = self.fea_agent.execute_code(csv_path = self.csv_path, code = code)

        ## Modeling Agent
        fea_eng_suggestions = fea_eng_result['Feature_analysis_suggestions']
        model_proposal = await self.modeling_agent.proposed_model(self.user_instructions, self.varb_info_path,fea_eng_suggestions, df_transformed, self.user_defined_target)
        modeling_code = await self.modeling_agent.generate_modeling_code(model_proposal)
        Model, evaluation_result,training_history = await self.modeling_agent.execute_code(df_transformed,  modeling_code)

        ## Evaluation agent
        self.evaluation_agent.user_instructions = user_instructions
        self.evaluation_agent.user_defined_target =user_defined_target
        evaluator_suggestion = await self.evaluation_agent.analyze_model(model_proposal, evaluation_result,training_history)


        ## Report agent
        self.report_agent.user_instructions = user_instructions
        self.report_agent.user_defined_target =user_defined_target
        optimization_suggestion = eval(evaluator_suggestion.replace("```", "").removeprefix("json"))['reasoning']
        report = await self.report_agent.generate_report(self.user_instructions, fea_eng_result, model_proposal, evaluation_result, optimization_suggestion)

        return report



