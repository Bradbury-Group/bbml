from bbml.foundations.gpt2.datamodels import GPTConfig
from bbml.core.datamodels.configs import TrainerConfig
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation
from bbml.foundations.gpt2.evaluation import GPT2FoundationLM

from lm_eval.evaluator import simple_evaluate
import itertools
import json

models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
tasks = ["lambada", "hellaswag", "wikitext", "c4"]

# Evaluate all combinations using itertools.product
for model, task in itertools.product(models, tasks):
    print(f"Evaluating {model} on {task}")
    
    # HuggingFace implementation
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model}",
        tasks=[task],
        batch_size=8,
        device="cuda:0",
        log_samples=False,
    )
    print(f"builtin hf - {model} on {task}")
    with open(f"{task}_{model}_hf_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # BBML implementation
    gpt_cfg = GPTConfig(from_hf=model)
    foundation = GPT2Foundation(gpt_cfg, train_config=None)
    foundation.to(device="cuda:0")
    lm = GPT2FoundationLM(foundation)
    
    
    results = simple_evaluate(
        model=lm,
        tasks=[task],
        batch_size=8,
        device=foundation.device,
        log_samples=False,
    )
    print(f"bbml impl - {model} on {task}")
    with open(f"{task}_{model}_bbml_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    