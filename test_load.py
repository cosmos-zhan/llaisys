from llaisys.models.qwen2 import Qwen2, LlaisysQwen2Meta
from llaisys.libllaisys import LIB_LLAISYS
import ctypes

print("Successfully imported Qwen2 and Linked Library")
print("Meta size:", ctypes.sizeof(LlaisysQwen2Meta))
print("Model Create function:", LIB_LLAISYS.llaisysQwen2ModelCreate)
print("Model Infer function:", LIB_LLAISYS.llaisysQwen2ModelInfer)
