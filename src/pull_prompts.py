"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


def pull_prompts_from_langsmith():
    """
    Faz pull do prompt do LangSmith Hub e salva localmente.
    """
    prompt = hub.pull("leonanluppi/bug_to_user_story_v1")
    
    # Assumindo que é um ChatPromptTemplate
    if hasattr(prompt, 'messages') and len(prompt.messages) >= 2:
        system_prompt = prompt.messages[0].prompt.template
        user_prompt = prompt.messages[1].prompt.template
    else:
        # Se for um PromptTemplate simples
        system_prompt = ""
        user_prompt = prompt.template
    
    data = {
        "bug_to_user_story_v1": {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": "v1",
            "created_at": "2025-01-15",
            "tags": ["bug-analysis", "user-story", "product-management"]
        }
    }
    
    save_path = Path("prompts/bug_to_user_story_v1.yml")
    save_yaml(data, str(save_path))


def main():
    """Função principal"""
    print_section_header("Pull de Prompts do LangSmith")
    
    required_vars = ["LANGSMITH_API_KEY"]
    if not check_env_vars(required_vars):
        return 1
    
    try:
        pull_prompts_from_langsmith()
        print("✅ Prompt baixado com sucesso!")
        return 0
    except Exception as e:
        print(f"❌ Erro ao baixar prompt: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
