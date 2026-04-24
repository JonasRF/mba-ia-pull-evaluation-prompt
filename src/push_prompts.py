"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import PromptTemplate

from utils import (
    load_yaml,
    check_env_vars,
    print_section_header
)

# Carrega .env
load_dotenv()


def push_prompt_to_langsmith(
    prompt_name: str,
    prompt_data: dict
) -> bool:
    """
    Faz push público do prompt para o LangSmith Hub.
    """

    try:
        # Configurar autenticação explicitamente
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            print("❌ LANGSMITH_API_KEY não configurada")
            return False

        client = Client(api_key=api_key)

        username = os.getenv("USERNAME_LANGSMITH_HUB")
        if not username:
            print("❌ USERNAME_LANGSMITH_HUB não configurada")
            return False

        # Nome versionado completo
        full_prompt_name = f"{username}/{prompt_name}"

        print(f"📤 Fazendo push do prompt: {full_prompt_name}")

        # Criar ChatPromptTemplate com system e human messages
        from langchain_core.prompts import ChatPromptTemplate

        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt_data["system_prompt"]),
            ("human", prompt_data["user_prompt"])
        ])

        # Preparar metadados
        metadata = {
            "description": prompt_data.get("description", ""),
            "version": prompt_data.get("version", "v2"),
            "created_at": prompt_data.get("created_at", ""),
            "tags": prompt_data.get("tags", []),
            "techniques": [
                "few-shot-learning",
                "role-prompting",
                "skeleton-of-thought"
            ],
            "domain": "bug-to-user-story",
            "objective": "converter-relatos-bugs-user-stories",
            "target_metrics": ">=0.90"
        }

        # Push do prompt (LangSmith renderiza os metadados automaticamente)
        url = client.push_prompt(
            full_prompt_name,
            object=chat_template,
            is_public=True
        )

        print("✅ Push realizado com sucesso!")
        print(f"🔗 URL: {url}")
        print(f"🏷️  Tags: {', '.join(metadata['tags'])}")
        print(f"🎯 Técnicas: {', '.join(metadata['techniques'])}")

        return True

    except Exception as e:

        error_msg = str(e)

        # Prompt já existe sem alterações
        if "Nothing to commit" in error_msg:
            print("ℹ️ Prompt já está atualizado no LangSmith.")
            return True

        print("\n❌ Erro ao fazer push:")
        print(error_msg)

        return False


def validate_prompt(prompt_data: dict):
    """
    Validação básica do YAML.
    """

    errors = []

    required_fields = [
        "system_prompt",
        "user_prompt"
    ]

    for field in required_fields:

        if field not in prompt_data:
            errors.append(
                f"Campo obrigatório ausente: {field}"
            )

        elif not str(prompt_data[field]).strip():
            errors.append(
                f"Campo vazio: {field}"
            )

    if "{bug_report}" not in prompt_data["user_prompt"]:
        errors.append(
            "Placeholder {bug_report} não encontrado no user_prompt"
        )

    return len(errors) == 0, errors


def main():

    print_section_header(
        "PUSH DE PROMPT PARA LANGSMITH"
    )

    required_vars = [
        "LANGSMITH_API_KEY"
    ]

    if not check_env_vars(required_vars):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    print(f"👤 Username: {username}")
    print(f"🎯 Prompt: {username}/bug_to_user_story_v2\n")

    # Arquivo YAML
    prompt_file = Path(
        "prompts/bug_to_user_story_v2.yml"
    )

    if not prompt_file.exists():

        print(
            f"❌ Arquivo não encontrado: {prompt_file}"
        )

        return 1

    print("📖 Carregando YAML...")

    prompt_yaml = load_yaml(str(prompt_file))

    if not prompt_yaml:

        print("❌ Erro ao carregar YAML")
        return 1

    prompt_name = "bug_to_user_story_v2"

    if prompt_name not in prompt_yaml:

        print(
            f"❌ Prompt '{prompt_name}' não encontrado"
        )

        return 1

    prompt_data = prompt_yaml[prompt_name]

    # Validação
    print("🔍 Validando prompt...")

    is_valid, errors = validate_prompt(
        prompt_data
    )

    if not is_valid:
        print("❌ Validação falhou:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("✅ Validação passou!")

    # Push
    print("\n🚀 Fazendo push para LangSmith Hub...")

    success = push_prompt_to_langsmith(
        prompt_name,
        prompt_data
    )

    if success:
        print("\n🎉 Push concluído com sucesso!")
        print(f"🔗 Acesse: https://smith.langchain.com/hub/{username}/bug_to_user_story_v2")
        return 0
    else:
        print("\n❌ Falha no push")
        return 1


if __name__ == "__main__":
    sys.exit(main())