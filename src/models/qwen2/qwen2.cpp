#include "qwen2.hpp"
#include "llaisys/ops.h"
// Include individual ops headers
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils/check.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath> // for sqrt
#include "../../llaisys/llaisys_tensor.hpp"

namespace llaisys {

// Helper to convert C handle to C++ Tensor
inline tensor_t to_cpp(llaisysTensor_t t) {
    if (!t) return nullptr;
    return reinterpret_cast<LlaisysTensor*>(t)->tensor;
}

// Helper to convert C array handle to C++ Tensor
inline tensor_t to_cpp(llaisysTensor_t* t_array, size_t idx) {
    if (!t_array || !t_array[idx]) return nullptr;
    return reinterpret_cast<LlaisysTensor*>(t_array[idx])->tensor;
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta& meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device_type(device), _device_id(device_id) {
    
    // Allocate weight arrays in the C struct
    _weights.attn_norm_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_q_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_q_b = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_k_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_k_b = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_v_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_v_b = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.attn_o_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_norm_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_gate_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_up_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));
    _weights.mlp_down_w = (llaisysTensor_t*)calloc(meta.nlayer, sizeof(llaisysTensor_t));

    init_buffers();
}

Qwen2Model::~Qwen2Model() {
    free(_weights.attn_norm_w);
    free(_weights.attn_q_w);
    free(_weights.attn_q_b);
    free(_weights.attn_k_w);
    free(_weights.attn_k_b);
    free(_weights.attn_v_w);
    free(_weights.attn_v_b);
    free(_weights.attn_o_w);
    free(_weights.mlp_norm_w);
    free(_weights.mlp_gate_w);
    free(_weights.mlp_up_w);
    free(_weights.mlp_down_w);
}

void Qwen2Model::init_buffers() {
    core::context().setDevice(_device_type, _device_id);

    // KV Cache
    // Shape: [nlayer, max_seq, n_kv_head, head_dim]
    // 为了简化，我们给每一层单独建立 KV
    for(size_t i=0; i<_meta.nlayer; ++i) {
        // K Cache
        _kv_caches.push_back({
            Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id),
            Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id)
        });
    }

    // Intermediate buffers
    // Assume batch size 1 for now as per infer interface
    _hidden_states = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _residual = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _ln_out = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _attn_out = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _mlp_out = Tensor::create({1, 1, _meta.hs}, _meta.dtype, _device_type, _device_id);
    
    // Logits: [1, vocab_size]
    _logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    
    // Pos Ids
    _pos_ids = Tensor::create({_meta.maxseq}, LLAISYS_DTYPE_I64, _device_type, _device_id);
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);
    
    // 1. Prepare Input
    // Re-create input tensor or load into buffer?
    // Prefill: ntoken > 1, Decode: ntoken = 1
    tensor_t input_tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_tokens->load(token_ids); // Copy from host

    // Prepare Pos Ids
    tensor_t current_pos_ids = _pos_ids->slice(0, 0, ntoken); // Just a view holder, need to fill data
    // Fill pos data
    std::vector<int64_t> pos_data(ntoken);
    for(size_t i=0; i<ntoken; ++i) pos_data[i] = _cur_pos + i;
    current_pos_ids->load(pos_data.data()); // Load pos ids

    // 2. Embedding
    // input_tokens: [seq]
    // in_embed: [vocab, hs]
    // hidden_states: [seq, hs]
    // Note: buffers defined in init are [1, 1, hs], need to resize for prefill
    // For simplicity, let's create dynamic views or reshape
    // Since ntoken can vary (1 during decode, N during prefill), buffers should adapt.
    // However, existing buffers are fixed size [1, 1, hs]. 
    // Let's re-allocate or reshape them properly.
    
    // Use 2D shape [seq, hs] for compatibility with Linear/Norm Ops
    std::vector<size_t> seq_shape = {ntoken, _meta.hs};
    
    // Embedding: [seq, hs]
    tensor_t hidden_states = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(hidden_states, input_tokens, to_cpp(_weights.in_embed));
    
    // No 3D View
    // hidden_states = hidden_states->view(seq_shape);
    // tensor_t residual = hidden_states; 
    
    // 3. Layers Loop
    for(size_t i=0; i<_meta.nlayer; ++i) {
        // --- Attention Block ---
        // Norm
        tensor_t normed = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(normed, hidden_states, to_cpp(_weights.attn_norm_w, i), _meta.epsilon);

        // QKV Projection
        // W_q: [n_head * head_dim, hidden_size] -> PyTorch Linear (OC, IC)
        // normed: [seq, hidden]
        // Q = normed * W_q.T
        // Q: [seq, n_head * head_dim]
        size_t q_dim = _meta.nh * _meta.dh;
        size_t k_dim = _meta.nkvh * _meta.dh;
        
        tensor_t q = Tensor::create({ntoken, q_dim}, _meta.dtype, _device_type, _device_id);
        tensor_t k = Tensor::create({ntoken, k_dim}, _meta.dtype, _device_type, _device_id);
        tensor_t v = Tensor::create({ntoken, k_dim}, _meta.dtype, _device_type, _device_id);

        ops::linear(q, normed, to_cpp(_weights.attn_q_w, i), to_cpp(_weights.attn_q_b, i));
        ops::linear(k, normed, to_cpp(_weights.attn_k_w, i), to_cpp(_weights.attn_k_b, i));
        ops::linear(v, normed, to_cpp(_weights.attn_v_w, i), to_cpp(_weights.attn_v_b, i));

        // Reshape for RoPE & Attention
        // [1, seq, n_head, head_dim]
        q = q->view({ntoken, _meta.nh, _meta.dh});
        k = k->view({ntoken, _meta.nkvh, _meta.dh});
        v = v->view({ntoken, _meta.nkvh, _meta.dh});

        // RoPE
        ops::rope(q, q, current_pos_ids, _meta.theta);
        ops::rope(k, k, current_pos_ids, _meta.theta);

        // KV Cache Update
        // _kv_caches[i].k: [max_seq, nkvh, dh]
        // Copy current k into cache at _cur_pos
        // For slice: dim=0, start=_cur_pos, end=_cur_pos+ntoken
        tensor_t k_cache_slot = _kv_caches[i].k->slice(0, _cur_pos, _cur_pos + ntoken);
        tensor_t v_cache_slot = _kv_caches[i].v->slice(0, _cur_pos, _cur_pos + ntoken);
        
        // We need a copy operator to copy data from k to k_cache_slot
        // Assuming tensor copy/load or rearrange can do it.
        // Or if ops::rearrange supports copying with different strides/shapes?
        // Actually, simple memcpy logic if contiguous.
        // Let's use `rearrange` if implemented, otherwise we need a copy op.
        // But wait, k and k_cache_slot might have different strides?
        // k: [seq, nkvh, dh] (contiguous packed)
        // k_cache_slot: view of large tensor. if large tensor is contiguous [max_seq, ...], then slice is contiguous too in memory layout logic but strided in larger context? No, slice of dim 0 preserves contiguousness IF it is row major.
        // Yes, dim 0 slice of contiguous tensor is contiguous.
        
        // But `rearrange` signature: void rearrange(out, in).
        // Let's assume we can use it for copy.
        ops::rearrange(k_cache_slot, k);
        ops::rearrange(v_cache_slot, v);

        // Prepare Full K/V for Attention (History + Current)
        // History depends on _cur_pos. 
        // total_len = _cur_pos + ntoken
        tensor_t k_full = _kv_caches[i].k->slice(0, 0, _cur_pos + ntoken);
        tensor_t v_full = _kv_caches[i].v->slice(0, 0, _cur_pos + ntoken);

        // Attention
        tensor_t attn_val = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        // scale = 1 / sqrt(head_dim)
        float scale = 1.0f / std::sqrt((float)_meta.dh);
        
        ops::self_attention(attn_val, q, k_full, v_full, scale);

        // Output Projection
        // Reshape attn_val back to [seq, hidden] (2D)
        // [seq, nhead, head_dim] -> [seq, nhead*head_dim]
        attn_val = attn_val->view({ntoken, _meta.hs});
        
        tensor_t attn_output = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::linear(attn_output, attn_val, to_cpp(_weights.attn_o_w, i), nullptr); 

        // Residual Add
        ops::add(hidden_states, hidden_states, attn_output);

        // --- MLP Block ---
        // Norm
        normed = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(normed, hidden_states, to_cpp(_weights.mlp_norm_w, i), _meta.epsilon);

        // Gate & Up
        // Gate & Up
        // [seq, intermediate]
        tensor_t gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        tensor_t up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        
        ops::linear(gate, normed, to_cpp(_weights.mlp_gate_w, i), nullptr);
        ops::linear(up, normed, to_cpp(_weights.mlp_up_w, i), nullptr);

        // SwiGLU
        // In-place or new tensor? Ops signature: out, gate, up. 
        // Let's use gate as output to save memory? SwiGLU might not be safe in-place if gate is needed.
        // Create new out.
        tensor_t swiglu_out = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::swiglu(swiglu_out, gate, up);

        // Down
        tensor_t mlp_output = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
        ops::linear(mlp_output, swiglu_out, to_cpp(_weights.mlp_down_w, i), nullptr);

        // Residual Add
        ops::add(hidden_states, hidden_states, mlp_output);
    }

    // 4. Final Norm
    tensor_t final_normed = Tensor::create(seq_shape, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(final_normed, hidden_states, to_cpp(_weights.out_norm_w), _meta.epsilon);

    // 5. Logits
    // Only need the last token's logits for next token generation
    // slice last token: [1, hidden] (dim 0 since it is 2D)
    tensor_t last_hidden = final_normed->slice(0, ntoken-1, ntoken); 
    // Is [1, hidden]
    // last_hidden = last_hidden->view({1, _meta.hs}); // Already [1, hidden]
    last_hidden = last_hidden->view({1, _meta.hs});

    tensor_t logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, last_hidden, to_cpp(_weights.out_embed), nullptr); // lm_head = in_embed usually? Or separate? Qwen2 usually ties weights? The struct has `out_embed`.

    // 6. Argmax
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    tensor_t max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id); // Dumy
    
    ops::argmax(max_idx, max_val, logits);

    // Get result to CPU
    int64_t next_token_id;
    // We need to copy max_idx data to host. 
    // If device is CPU, just read.
    if (_device_type == LLAISYS_DEVICE_CPU) {
        next_token_id = *reinterpret_cast<int64_t*>(max_idx->data());
    } else {
        // Copy back
        // For now assume CPU
        next_token_id = 0; 
        // TODO: Implement copy D2H
    }

    // Update pos
    _cur_pos += ntoken;

    return next_token_id;
}

} // namespace llaisys
