from bbml.registries import WeightExtractorRegistry, MetricRegistry
from bbml.analysis.extractors.gpt2 import GPT2WeightExtractor
from bbml.analysis.extractors.llama import LlamaWeightExtractor
from bbml.analysis.metrics.cosine import CosineMetric


WeightExtractorRegistry.register("gpt2")(GPT2WeightExtractor)
WeightExtractorRegistry.register("llama")(LlamaWeightExtractor)
MetricRegistry.register("cosine")(CosineMetric)
