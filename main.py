from llm.azure_openai_client import AzureOpenAIClient
from parser.resume_parser import ResumeParser
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv(override=True)
resume_path = Path("C:/Users/adity/project/personal-resume-server/resume/output.txt")

deployment_name=os.getenv("DEPLOYMENT_NAME", "your-deployment-name")
azure_openai_base_url=os.getenv("AZURE_API_BASE", "https://your-resource-name.openai.azure.com/")
azure_api_key=os.getenv("AZURE_API_KEY", "your-api-key")
api_version=os.getenv("AZURE_API_VERSION", "2023-06-01-preview")
llm_client = AzureOpenAIClient(
    azure_endpoint=azure_openai_base_url,
    api_version=api_version,
    api_key=azure_api_key,
    model=deployment_name
)
resume_parser = ResumeParser(llm_client)

with open(resume_path, "r") as f:
    resume_text = f.read()

resume = resume_parser.parse(resume_text=resume_text)
print(resume.model_dump_json(indent=4))