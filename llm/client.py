import os
import google.generativeai as genai
from llm.tools import get_feature_importance, predict_risk, get_model_metrics

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Define tools for Gemini
tools = [get_feature_importance, predict_risk, get_model_metrics]

SYSTEM_INSTRUCTION = """
Você é um assistente especializado em análise de risco cardíaco, integrado a um modelo de 
Machine Learning treinado com dados clínicos reais de 918 pacientes.

Você tem acesso a três ferramentas:
- get_feature_importance: retorna quais variáveis clínicas mais influenciam o risco cardíaco
- predict_risk: faz uma predição de risco para um paciente com base em seus dados clínicos
- get_model_metrics: retorna as métricas de desempenho do modelo (Recall, F1, Acurácia)

Regras:
- Sempre use as ferramentas disponíveis para responder com dados reais, não invente números
- Explique os resultados em linguagem clara, acessível para não-especialistas
- Quando fizer uma predição, explique o que o resultado significa clinicamente
- Lembre sempre que esta ferramenta é um apoio à triagem, não substitui diagnóstico médico
- Responda sempre em português
"""


def create_chat():
    """Cria uma nova sessão de chat com o Gemini."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=tools,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = model.start_chat(enable_automatic_function_calling=True)
    return chat


def send_message(chat, message: str) -> str:
    """Envia mensagem ao chat e retorna a resposta."""
    try:
        response = chat.send_message(message)
        return response.text
    except Exception as e:
        return f"Erro ao processar mensagem: {str(e)}"