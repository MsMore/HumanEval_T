
# Combinatorial Benchmark HumanEval_T

This repository contains the implementation and evaluation framework for the paper titled "Addressing Data Leakage in HumanEval Using
Combinatorial Test Design"

## Repository Structure

```
combinatorial-benchmark/
├── humanEval benchmark/           # HumanEval-style programming challenges
│   ├── prompts/                  # Individual problem definitions
│   │   └── problem[1-10].json    # Problem specifications and test cases
│   ├── generated_outputs_problem[1-10].txt  # Model outputs for each problem
│   ├── huggingfaceBenchmark.ipynb  # Jupyter notebook for HuggingFace model evaluation
│   └── human.py                  # Human baseline implementation
├── meta benchmark/               # Meta-learning benchmark tasks
│   ├── meta_prompts/            # Meta-learning problem definitions
│   │   └── problem_[1-10].json  # Meta-problem specifications
│   └── generated_outputs_problem[1-10].txt  # Model outputs for meta-problems
├── main.py                      # Main evaluation script
└── README.md                    # This file
```

## Components

### HumanEval Benchmark
- Contains 10 programming problems from the HumanEval benchmark
- Each problem in `prompts/` includes:
  - Problem description
  - Input/output specifications
  - Test cases
  - Evaluation criteria
- Generated outputs are stored in separate files for analysis
- Includes a Jupyter notebook for evaluating HuggingFace models
- `human.py` provides code to evaluate the humanEval dataset samples

### Meta Benchmark
- Features 10 meta-learning problems to assess LLMs' ability to generalize across problem patterns
- Problems in `meta_prompts/` include:
  - Meta-pattern descriptions
  - Instance generation rules
  - Evaluation metrics
- Generated outputs capture model responses for meta-learning tasks

### Main Script
The `main.py` script provides functionality for:
- Loading and processing problem definitions
- Interfacing with different LLM providers (OpenAI, Anthropic, Ollama)
- Running evaluations and collecting results
- Testing generated solutions against provided test cases

## Usage

1. Install dependencies:
```bash
pip install openai anthropic requests
```

2. Set up API keys in environment variables:
```bash
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

3. Run evaluations:
```bash
python main.py
```
