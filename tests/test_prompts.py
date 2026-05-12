"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE_V2 = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class TestPrompts:
    def setup_method(self):
        data = load_prompts(str(PROMPT_FILE_V2))
        self.prompt_data = data["bug_to_user_story_v2"]

    def test_prompt_has_system_prompt(self):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in self.prompt_data, (
            "Campo 'system_prompt' não encontrado no prompt"
        )
        assert self.prompt_data["system_prompt"].strip(), (
            "O campo 'system_prompt' está vazio"
        )


    def test_prompt_has_role_definition(self):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = self.prompt_data["system_prompt"]
        role_indicators = ["Você é um", "Você é uma", "Você é o", "You are a", "You are an"]
        assert any(indicator in system_prompt for indicator in role_indicators), (
            "Prompt não define uma persona/role. "
            "Esperado algo como 'Você é um Senior Product Manager'"
        )

    def test_prompt_mentions_format(self):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = self.prompt_data["system_prompt"]
        format_keywords = [
            "User Story", "user story",
            "Como um", "Como uma",
            "Critérios de Aceitação",
            "Markdown", "markdown",
        ]
        assert any(kw in system_prompt for kw in format_keywords), (
            "Prompt não menciona formato de saída esperado (User Story, Markdown, etc.)"
        )

    def test_prompt_has_few_shot_examples(self):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = self.prompt_data["system_prompt"]
        has_example_marker = "EXEMPLO" in system_prompt
        has_input_marker = "ENTRADA:" in system_prompt
        has_output_marker = "SAÍDA CORRETA:" in system_prompt or "SAÍDA:" in system_prompt
        assert has_example_marker and has_input_marker and has_output_marker, (
            "Prompt não contém exemplos few-shot no formato EXEMPLO / ENTRADA / SAÍDA"
        )

    def test_prompt_no_todos(self):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        system_prompt = self.prompt_data.get("system_prompt", "")
        user_prompt = self.prompt_data.get("user_prompt", "")
        full_text = system_prompt + user_prompt
        assert "[TODO]" not in full_text, (
            "Prompt contém marcadores '[TODO]' não resolvidos"
        )

    def test_minimum_techniques(self):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = self.prompt_data.get(
            "techniques_applied",
            self.prompt_data.get("techniques_used", [])
        )
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
