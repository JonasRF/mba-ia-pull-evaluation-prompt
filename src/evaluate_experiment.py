"""
Evaluate prompts using LangSmith Experiments.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

from langsmith import Client
from langsmith.evaluation import evaluate

from langchain import hub

from utils import (
    check_env_vars,
    print_section_header,
    get_llm as get_configured_llm,
)

from metrics import (
    evaluate_f1_score,
    evaluate_clarity,
    evaluate_precision,
)

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

PROJECT_NAME = os.getenv(
    "LANGSMITH_PROJECT",
    "prompt-evaluation-project"
)

DATASET_FILE = "datasets/bug_to_user_story.jsonl"


# ============================================================
# LLM
# ============================================================

def get_llm():
    return get_configured_llm(temperature=0)


# ============================================================
# DATASET
# ============================================================

def load_dataset_from_jsonl(
    jsonl_path: str
) -> List[Dict[str, Any]]:

    examples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def create_or_get_dataset(
    client: Client,
    dataset_name: str,
    jsonl_path: str
):

    print(f"📦 Preparando dataset: {dataset_name}")

    datasets = list(
        client.list_datasets(dataset_name=dataset_name)
    )

    if datasets:
        print("✅ Dataset já existe")
        return datasets[0]

    dataset = client.create_dataset(
        dataset_name=dataset_name
    )

    examples = load_dataset_from_jsonl(jsonl_path)

    for example in examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs=example["inputs"],
            outputs=example["outputs"]
        )

    print(f"✅ Dataset criado com {len(examples)} exemplos")

    return dataset


# ============================================================
# PROMPT
# ============================================================

def pull_prompt(prompt_name: str):

    print(f"📥 Pull do prompt: {prompt_name}")

    prompt = hub.pull(prompt_name)

    print("✅ Prompt carregado com sucesso")

    return prompt


# ============================================================
# TARGET
# ============================================================

def build_target(prompt):

    llm = get_llm()

    chain = prompt | llm

    def target(inputs: Dict[str, Any]):

        response = chain.invoke(inputs)

        return {
            "answer": response.content
        }

    return target


# ============================================================
# EVALUATORS
# ============================================================

def f1_evaluator(run, example):

    prediction = run.outputs.get("answer", "")

    reference = example.outputs.get(
        "reference",
        ""
    )

    question = example.inputs.get(
        "bug_report",
        ""
    )

    result = evaluate_f1_score(
        question,
        prediction,
        reference
    )

    return {
        "key": "f1_score",
        "score": result["score"]
    }


def clarity_evaluator(run, example):

    prediction = run.outputs.get("answer", "")

    reference = example.outputs.get(
        "reference",
        ""
    )

    question = example.inputs.get(
        "bug_report",
        ""
    )

    result = evaluate_clarity(
        question,
        prediction,
        reference
    )

    return {
        "key": "clarity",
        "score": result["score"]
    }


def precision_evaluator(run, example):

    prediction = run.outputs.get("answer", "")

    reference = example.outputs.get(
        "reference",
        ""
    )

    question = example.inputs.get(
        "bug_report",
        ""
    )

    result = evaluate_precision(
        question,
        prediction,
        reference
    )

    return {
        "key": "precision",
        "score": result["score"]
    }


def helpfulness_evaluator(run, example):

    prediction = run.outputs.get("answer", "")

    reference = example.outputs.get(
        "reference",
        ""
    )

    question = example.inputs.get(
        "bug_report",
        ""
    )

    clarity = evaluate_clarity(
        question,
        prediction,
        reference
    )["score"]

    precision = evaluate_precision(
        question,
        prediction,
        reference
    )["score"]

    helpfulness = (
        clarity + precision
    ) / 2

    return {
        "key": "helpfulness",
        "score": round(helpfulness, 4)
    }


def correctness_evaluator(run, example):

    prediction = run.outputs.get("answer", "")

    reference = example.outputs.get(
        "reference",
        ""
    )

    question = example.inputs.get(
        "bug_report",
        ""
    )

    f1 = evaluate_f1_score(
        question,
        prediction,
        reference
    )["score"]

    precision = evaluate_precision(
        question,
        prediction,
        reference
    )["score"]

    correctness = (
        f1 + precision
    ) / 2

    return {
        "key": "correctness",
        "score": round(correctness, 4)
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print_section_header(
        "AVALIAÇÃO DE PROMPTS COM LANGSMITH"
    )

    required_vars = [
        "LANGSMITH_API_KEY",
        "OPENAI_API_KEY",
        "USERNAME_LANGSMITH_HUB"
    ]

    if not check_env_vars(required_vars):
        return 1

    client = Client()

    dataset_name = f"{PROJECT_NAME}-eval"

    dataset = create_or_get_dataset(
        client=client,
        dataset_name=dataset_name,
        jsonl_path=DATASET_FILE
    )

    username = os.getenv(
        "USERNAME_LANGSMITH_HUB"
    )

    prompt_name = (
        f"{username}/bug_to_user_story_v2"
    )

    prompt = pull_prompt(prompt_name)

    target = build_target(prompt)

    experiment_prefix = (
        "bug-to-user-story-evaluation"
    )

    print("\n🚀 Iniciando avaliação...\n")

    results = evaluate(
        target,
        data=dataset.name,
        evaluators=[
            f1_evaluator,
            clarity_evaluator,
            precision_evaluator,
            helpfulness_evaluator,
            correctness_evaluator
        ],
        experiment_prefix=experiment_prefix,
    )

    print("\n✅ Avaliação concluída!")

    print("\n📊 Experimento criado no LangSmith")

    print(
        f"\n🔗 https://smith.langchain.com/projects/p/{PROJECT_NAME}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())