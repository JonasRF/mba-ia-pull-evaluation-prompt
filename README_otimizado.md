# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

> Projeto desenvolvido como parte do MBA em Inteligência Artificial.
> Repositório: https://github.com/JonasRF/mba-ia-pull-evaluation-prompt.git

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Stack e Tecnologias](#stack-e-tecnologias)
- [Pré-requisitos](#pré-requisitos)
- [Como Executar](#como-executar)
- [Técnicas Aplicadas](#técnicas-aplicadas)
- [Resultados Obtidos](#resultados-obtidos)
- [Evidências no LangSmith](#evidências-no-langsmith)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Dicas Finais](#dicas-finais)

---

## Sobre o Projeto

Este projeto consiste em um pipeline completo de **engenharia de prompts**, com três etapas principais:

1. **Pull** de prompts de baixa qualidade do LangSmith Prompt Hub
2. **Otimização** desses prompts aplicando técnicas avançadas de Prompt Engineering
3. **Avaliação** da qualidade via métricas customizadas, com meta mínima de **0.9 (90%)** em todas elas

### Critério de Aprovação

| Métrica | Mínimo |
|---|---|
| Helpfulness | >= 0.9 |
| Correctness | >= 0.9 |
| F1-Score | >= 0.9 |
| Clarity | >= 0.9 |
| Precision | >= 0.9 |

> **IMPORTANTE:** TODAS as 5 métricas devem atingir >= 0.9 individualmente, não apenas a média.

---

## Stack e Tecnologias

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.9+ |
| Framework | LangChain |
| Avaliação | LangSmith |
| Gestão de Prompts | LangSmith Prompt Hub |
| Formato de Prompts | YAML |
| LLM (opção 1) | OpenAI `gpt-4o-mini` / `gpt-4o` |
| LLM (opção 2) | Google Gemini `gemini-2.5-flash` (gratuito) |

---

## Pré-requisitos

Antes de executar o projeto, você precisará:

- **Python 3.9+** instalado
- Conta e API Key no **LangSmith**: https://smith.langchain.com
- API Key de pelo menos **um** dos provedores de LLM:
  - **OpenAI**: https://platform.openai.com/api-keys — custo estimado ~$1–5
  - **Google Gemini** (gratuito): https://aistudio.google.com/app/apikey — limite de 15 req/min, 1500 req/dia

---

## Como Executar

### 1. Clone o repositório e configure o ambiente

```bash
git clone https://github.com/JonasRF/mba-ia-pull-evaluation-prompt.git
cd mba-ia-pull-evaluation-prompt
```

Crie e ative o ambiente virtual:

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

### 2. Configure as variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env` com os valores abaixo:

```env
# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=sua_api_key_aqui
LANGSMITH_PROJECT=nome_do_seu_projeto

# Para descobrir seu username: publique qualquer prompt no LangSmith Hub,
# abra-o e clique no ícone de cadeado (🔒) para ver seu username.
USERNAME_LANGSMITH_HUB=seu_username_aqui

# OpenAI (se for usar OpenAI)
OPENAI_API_KEY=sua_chave_aqui

# Google Gemini (se for usar Gemini)
GOOGLE_API_KEY=sua_chave_aqui

# Escolha o provedor (descomente apenas um bloco)
# --- OpenAI ---
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# EVAL_MODEL=gpt-4o

# --- Google Gemini ---
# LLM_PROVIDER=google
# LLM_MODEL=gemini-2.5-flash
# EVAL_MODEL=gemini-2.5-flash
```

> A escolha do provedor de LLM fica a seu critério.

---

### 3. Faça o pull do prompt base (v1)

```bash
python src/pull_prompts.py
```

Isso irá baixar o prompt de baixa qualidade `leonanluppi/bug_to_user_story_v1` e salvá-lo localmente em `prompts/bug_to_user_story_v1.yml`.

---

### 4. Otimize o prompt (v2)

Edite o arquivo `prompts/bug_to_user_story_v2.yml` aplicando as técnicas de Prompt Engineering. Veja a seção [Técnicas Aplicadas](#técnicas-aplicadas) para entender o que foi feito neste projeto.

---

### 5. Faça o push do prompt otimizado

```bash
python src/push_prompts.py
```

Isso publica o prompt otimizado como `{seu_username}/bug_to_user_story_v2` no LangSmith Hub.

---

### 6. Execute a avaliação

```bash
python src/evaluate.py
```

Exemplo de saída esperada após a otimização:

```
==================================================
Prompt: {seu_username}/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 0.94 ✓
  - Correctness: 0.96 ✓

Métricas Base:
  - F1-Score: 0.93 ✓
  - Clarity: 0.95 ✓
  - Precision: 0.92 ✓

✅ STATUS: APROVADO - Todas as métricas >= 0.9
```

> Caso alguma métrica fique abaixo de 0.9, ajuste o prompt, faça push novamente e repita a avaliação. Espera-se entre 3 e 5 iterações até aprovação.

---

### 7. Execute os testes de validação

```bash
pytest tests/test_prompts.py
```

---

## Técnicas Aplicadas

Para criar o prompt `bug_to_user_story_v2`, foram aplicadas três técnicas avançadas de Prompt Engineering:

---

### 1. Role Prompting

**O que é:** Atribuir uma identidade e especialidade explícita ao modelo antes de qualquer instrução.

**Por que foi escolhida:** Sem um papel definido, o modelo tende a gerar user stories genéricas. Ao declarar um Senior PM com domínios específicos, o output alinha vocabulário, nível de detalhe e estrutura ao padrão profissional esperado.

**Como foi aplicada:**

```
Você é um Senior Product Manager especialista em engenharia de requisitos ágil.
Domínios: e-commerce, SaaS B2B, ERP, CRM, mobile apps.
Missão: replicar EXATAMENTE a estrutura, seções, wording e nível de detalhe dos exemplos abaixo.
```

O papel vem acompanhado de uma missão restritiva — *"fidelidade total ao padrão, não criatividade"* — para evitar que o modelo improvise fora do template.

---

### 2. Chain of Thought (CoT)

**O que é:** Instruir o modelo a raciocinar em etapas antes de produzir o output, sem escrever esse raciocínio na resposta final.

**Por que foi escolhida:** Bugs têm complexidades diferentes. Sem uma etapa de classificação prévia, o modelo pode aplicar a estrutura errada (ex: usar seções de COMPLEX num bug SIMPLE). O CoT força essa decisão antes de escrever.

**Como foi aplicada:**

O modelo é instruído a identificar mentalmente três coisas, nessa ordem:
1. A complexidade do bug (SIMPLE / MEDIUM / COMPLEX)
2. A persona correta para aquela complexidade
3. As seções obrigatórias para aquele nível

![Chain of Thought aplicado](https://github.com/user-attachments/assets/3df607b9-68d7-4450-ace9-54b84621b42e)

A instrução *"analise internamente, NÃO escreva na resposta"* é o que transforma CoT em raciocínio latente, mantendo o output limpo.

---

### 3. Few-Shot Learning (15 exemplos calibrados)

**O que é:** Fornecer pares `ENTRADA → SAÍDA CORRETA` como exemplos dentro do próprio prompt, para que o modelo aprenda o padrão por indução.

**Por que foi escolhida:** É a técnica de maior impacto para tarefas com formato rígido. Em vez de descrever a estrutura em regras abstratas, os exemplos mostram o padrão concreto — o modelo generaliza por similaridade estrutural.

**Como foi aplicada:**

Os 15 exemplos foram distribuídos para cobrir todas as combinações relevantes:

![Distribuição dos exemplos](https://github.com/user-attachments/assets/b7e2965f-393a-4450-9046-cf6b36dd18d6)

Cada exemplo ensina três coisas simultaneamente: qual persona usar, quais seções incluir e qual profundidade de detalhe aplicar. O modelo não precisa inferir regras — ele encontra o exemplo mais similar e replica a estrutura, substituindo apenas o conteúdo.

---

### Como as três técnicas se complementam

| Técnica | Papel no sistema |
|---|---|
| Role Prompting | Define **quem** responde e com qual mentalidade |
| Chain of Thought | Define **como** raciocinar antes de escrever |
| Few-Shot Learning | Define **o quê** escrever, com exemplos concretos |

Juntas, eliminam os três principais pontos de falha em geração de user stories: persona errada, estrutura inadequada para a complexidade do bug e nível de detalhe inconsistente.

---

## Resultados Obtidos

### Tabela comparativa: v1 (ruim) vs v2 (otimizado)

![Tabela comparativa v1 vs v2](https://github.com/user-attachments/assets/5acd383a-f466-417b-8087-96a500f5df15)

### Métricas finais — todas >= 0.9 ✅

![Screenshot das métricas aprovadas](https://github.com/user-attachments/assets/e4d4fd79-0975-401a-8b1f-911b3ddce82d)

### Dashboard público no LangSmith

🔗 https://smith.langchain.com/public/8e114d59-4c04-4a2e-bb74-c907a62a2d0a/d

---

## Evidências no LangSmith

### Painel principal de tracing

![Painel de tracing LangSmith](https://github.com/user-attachments/assets/70646b7c-7d02-4adb-866a-32a0505d2858)

![Detalhe dos exemplos trabalhados](https://github.com/user-attachments/assets/45c214e4-8252-4bac-af47-4023ece723db)

### Links públicos dos exemplos avaliados

- [Exemplo 01](https://smith.langchain.com/public/43571862-3de4-4d6d-903b-88336a0106a7/r)
- [Exemplo 02](https://smith.langchain.com/public/42d38062-4e1b-4280-a9e9-e27921121b5f/r)
- [Exemplo 03](https://smith.langchain.com/public/d9abf67d-2d8b-4445-a4bc-0ddc8a1c23f4/r)

---

## Estrutura do Projeto

```
mba-ia-pull-evaluation-prompt/
├── .env.example                      # Template das variáveis de ambiente
├── requirements.txt                  # Dependências Python
├── README.md                         # Este arquivo
│
├── prompts/
│   ├── bug_to_user_story_v1.yml      # Prompt inicial baixado do LangSmith
│   └── bug_to_user_story_v2.yml      # Prompt otimizado (criado neste projeto)
│
├── datasets/
│   └── bug_to_user_story.jsonl       # 15 exemplos de bugs (5 simples, 7 médios, 3 complexos)
│
├── src/
│   ├── pull_prompts.py               # Script de pull do LangSmith
│   ├── push_prompts.py               # Script de push para o LangSmith
│   ├── evaluate.py                   # Script de avaliação (não alterar)
│   ├── metrics.py                    # 5 métricas implementadas (não alterar)
│   └── utils.py                      # Funções auxiliares (não alterar)
│
└── tests/
    └── test_prompts.py               # Testes de validação com pytest
```

---

## Dicas Finais

- **Especificidade e contexto** fazem toda a diferença na refatoração de prompts
- **Few-Shot Learning com exemplos bem calibrados** é a técnica de maior impacto para tarefas com formato rígido
- **Chain of Thought** é essencial para bugs com complexidades variadas — força a classificação antes de escrever
- **Use o Tracing do LangSmith** como principal ferramenta de debug — ele mostra exatamente o que o LLM está processando
- **Não altere os datasets de avaliação** — apenas os prompts em `prompts/bug_to_user_story_v2.yml`
- **Itere entre 3 e 5 vezes** — é o esperado para atingir 0.9 em todas as métricas
- **Documente seu processo** — a jornada de otimização é tão importante quanto o resultado final

---

## Links Úteis

- [Repositório do projeto](https://github.com/JonasRF/mba-ia-pull-evaluation-prompt.git)
- [Repositório boilerplate do desafio](https://github.com/devfullcycle/mba-ia-prompt-engineering)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
