#pragma once
#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include <vector>
#include <memory> 

namespace llaisys {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta& meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model();

    // 获取权重结构的指针，供 Python 填充数据
    LlaisysQwen2Weights* getWeights() { return &_weights; }
    
    // 或者提供一个初始化权重的接口，这里为了简单直接暴露结构体
    // 但必须确保并在 infer 前所有权重都被正确初始化

    // 推理接口
    // 输入: token_ids (当前步的输入 token), ntoken (通常为 1，除了 prefill 阶段)
    // 返回: 下一个 token id
    int64_t infer(const int64_t* token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights; // 存储权重的 Tensor 对象 (shared_ptr)
    
    llaisysDeviceType_t _device_type;
    int _device_id;

    // KV Cache
    // Layer -> [K_Cache, V_Cache]
    struct KVCache {
        tensor_t k;
        tensor_t v;
    };
    std::vector<KVCache> _kv_caches;

    // 用于推理的当前位置
    size_t _cur_pos = 0;

    // 预分配的中间变量 (Buffer) 以避免重复 malloc
    tensor_t _hidden_states; // [1, 1, hidden_size] or [1, seq, hidden_size]
    tensor_t _residual;
    tensor_t _ln_out;        // Norm output
    tensor_t _attn_out;      
    tensor_t _mlp_out;
    tensor_t _logits;

    // 辅助 Tensor
    tensor_t _tokens_tensor; // 用于存放输入的 token_ids
    tensor_t _pos_ids;       // 用于 RoPE

    // 初始化 KV Cache 和 Buffer
    void init_buffers();
    
    // 如果权重是裸指针数组，我们需要分配它们对应的 Tensor 对象容器
    // 注意：LlaisysQwen2Weights 里定义的是 llaisysTensor_t* (也就是 void**)
    // 我们需要在 C++ 端管理这些数组的内存
    std::vector<tensor_t> _layers_attn_norm_w;
    std::vector<tensor_t> _layers_attn_q_w;
    std::vector<tensor_t> _layers_attn_q_b;
    std::vector<tensor_t> _layers_attn_k_w;
    std::vector<tensor_t> _layers_attn_k_b;
    std::vector<tensor_t> _layers_attn_v_w;
    std::vector<tensor_t> _layers_attn_v_b;
    std::vector<tensor_t> _layers_attn_o_w;
    std::vector<tensor_t> _layers_mlp_norm_w;
    std::vector<tensor_t> _layers_mlp_gate_w;
    std::vector<tensor_t> _layers_mlp_up_w;
    std::vector<tensor_t> _layers_mlp_down_w;

    void allocate_layers_weights();
};

} // namespace llaisys
