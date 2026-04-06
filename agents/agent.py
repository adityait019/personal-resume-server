#agent.py
from datetime import datetime

from google.adk.agents.llm_agent import Agent
import os
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from agents.agent_executor import PersonalAgentExecutor
from agents.agent_card_config import build_agent_card
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
# Make your agent A2A-compatible
import logging

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(
    level=logging.DEBUG,filemode='a',filename="root_agent.log",
    format="CorteX:%(asctime)s - %(levelname)s - %(message)s"
)


async def about_professional_life() -> str:
    return "I am a software engineer with experience in building web applications and machine learning models. I have worked on various projects involving natural language processing and data analysis. I am passionate about learning new technologies and improving my skills."

async def about_personal_life() -> str:
    return "In my personal life, I enjoy hiking, cooking, and spending time with my family. I also have a keen interest in photography and often go on photo walks during weekends."    
DEPLOYMENT_NAME=os.environ["DEPLOYMENT_NAME"]
AZURE_API_KEY=os.environ['AZURE_API_KEY']
AZURE_API_BASE=os.environ['AZURE_API_BASE']
AZURE_API_VERSION=os.environ['AZURE_API_VERSION']
MODEL=f"azure/{DEPLOYMENT_NAME}"
llm = LiteLlm(model=MODEL,
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION)





root_agent = Agent(
    name='aditya_personal_assistant',
    model=llm,
    description='An assistant designed to help with various tasks.',
    instruction="""
    You are Aditya's personal assistant. You can answer questions about Aditya's professional and personal life.  
    Always be helpful and provide accurate information based on what you know about Aditya.
""".strip(),
    tools=[about_professional_life, about_personal_life,]
)


def create_runner() ->Runner:
    return Runner(
        app_name=root_agent.name,
        agent=root_agent,
        session_service=InMemorySessionService(),
    )

def build_app(host: str = '192.168.1.5', port: int = 8005) -> Starlette:

    task_store = InMemoryTaskStore()
    agent_executor = PersonalAgentExecutor(runner_or_factory=create_runner)
    request_handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)

    agent_card= build_agent_card()
    app=Starlette()
    a2a_app = A2AStarletteApplication(agent_card=agent_card,http_handler=request_handler)

    a2a_app.add_routes_to_app(app)

    @app.route("/health", methods=["GET"])
    @app.route("/healthz", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok",
                             "timestamp": datetime.now().astimezone().isoformat(),
                             "agent": root_agent.name})
    
    return app


app = build_app()

