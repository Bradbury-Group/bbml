from bbml.foundations.gpt2.datamodels import GPTConfig
from bbml.core.datamodels.configs import TrainerConfig
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation
from bbml.foundations.gpt2.evaluation import BaseFoundationLM

from lm_eval.evaluator import simple_evaluate

# 1. Build your bbml foundation
gpt_cfg = GPTConfig()                     # or customized config
trainer_cfg = TrainerConfig()             # or None if you don't need it
foundation = GPT2Foundation(gpt_cfg, trainer_cfg)

# 2. Wrap it as an LM for lm-eval
lm = BaseFoundationLM(foundation)

# 3. Run evaluation
results = simple_evaluate(
    model=lm,                 # pass the LM instance, not a string
    tasks=["lambdada"],      # or any configured tasks
    num_fewshot=0,
    batch_size=1,
)