import os
from ml_agents import feature_analysis_Agent,ModelingAgent
import asyncio







# Entrypoint for the financial bot example.
# Run this as `python -m examples.financial_research_agent.main` and enter a
# financial research query, for example:
# "Write up an analysis of Apple Inc.'s most recent quarter."
async def main() -> None:
    # query = input("Enter a financial research query: ")
    # mgr = FinancialResearchManager()
    # await mgr.run(query)

    ## target provided
    csv_path  = input("Enter train data path: ")
    varb_info_path=input("Enter variable information path: ")
    user_instructions = input("Enter user instructions / problem statement: ")
    user_defined_target = input ("Enter prediction target: ")

    ## Feature analysis agent
    fea_agent = feature_analysis_Agent(user_instructions = user_instructions, user_defined_target=user_defined_target)
    fea_eng_result = await fea_agent.run(csv_path = csv_path, varb_info_path=varb_info_path)

    code = await fea_agent.generate_transformation_code(
        fea_eng_result['raw_profile'],
        fea_eng_result['Feature_analysis_suggestions']
    )
    df_transformed = fea_agent.execute_code(csv_path = csv_path, code = code)

    ## Modeling Agent
    modeling_agent=ModelingAgent()

    fea_eng_suggestions = fea_eng_result['Feature_analysis_suggestions']
    model_proposal = await modeling_agent.proposed_model(user_instructions, varb_info_path,fea_eng_suggestions, df_transformed, user_defined_target)
    









if __name__ == "__main__":
    asyncio.run(main())