from bbml.analysis.extractors.gpt2 import GPT2WeightExtractor
from bbml.analysis.extractors.llama import LlamaWeightExtractor
from bbml.analysis.extractors.opt import OPTWeightExtractor
from bbml.analysis.extractors.phi3 import Phi3WeightExtractor
from bbml.analysis.extractors.pythia import PythiaWeightExtractor
from bbml.analysis.metrics.cosine import CosineMetric
from bbml.registries import MetricRegistry, WeightExtractorRegistry

WeightExtractorRegistry.register("gpt2")(GPT2WeightExtractor)
WeightExtractorRegistry.register("llama")(LlamaWeightExtractor)
WeightExtractorRegistry.register("qwen")(LlamaWeightExtractor)
WeightExtractorRegistry.register("gemma2")(LlamaWeightExtractor)
WeightExtractorRegistry.register("deepseek")(LlamaWeightExtractor)
WeightExtractorRegistry.register("tinyllama")(LlamaWeightExtractor)
WeightExtractorRegistry.register("pythia")(PythiaWeightExtractor)
WeightExtractorRegistry.register("phi3")(Phi3WeightExtractor)
WeightExtractorRegistry.register("opt")(OPTWeightExtractor)
MetricRegistry.register("cosine")(CosineMetric)
