#pragma once
// trt_compat.hpp — TensorRT 10.x API（仅支持 JetPack 6.x / Orin）

#include <NvInfer.h>

// ── 内存池 / Workspace ────────────────────────────────────────────────────────
inline void trt_set_workspace(nvinfer1::IBuilderConfig * cfg, size_t size)
{
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size);
}

// ── Tensor 名称查询 ───────────────────────────────────────────────────────────
inline const char * trt_get_tensor_name(nvinfer1::ICudaEngine * engine, int index)
{
  return engine->getIOTensorName(index);
}

// ── Tensor 形状查询（engine 静态形状）────────────────────────────────────────
inline nvinfer1::Dims trt_get_tensor_shape(nvinfer1::ICudaEngine * engine,
                                            const char * name, int /*index*/)
{
  return engine->getTensorShape(name);
}

// ── Tensor 形状查询（context 运行时形状）─────────────────────────────────────
inline nvinfer1::Dims trt_get_context_tensor_shape(nvinfer1::IExecutionContext * ctx,
                                                    const char * name, int /*index*/)
{
  return ctx->getTensorShape(name);
}

// ── 动态输入形状设置 ──────────────────────────────────────────────────────────
inline bool trt_set_input_shape(nvinfer1::IExecutionContext * ctx,
                                 const char * name, int /*index*/,
                                 const nvinfer1::Dims & dims)
{
  return ctx->setInputShape(name, dims);
}

// ── Tensor 数据类型查询 ───────────────────────────────────────────────────────
inline nvinfer1::DataType trt_get_tensor_dtype(nvinfer1::ICudaEngine * engine,
                                                const char * name, int /*index*/)
{
  return engine->getTensorDataType(name);
}

// ── Buffer 地址绑定 ───────────────────────────────────────────────────────────
inline void trt_set_tensor_address(nvinfer1::IExecutionContext * ctx,
                                    const char * name, int /*index*/,
                                    void * ptr)
{
  ctx->setTensorAddress(name, ptr);
}

// ── 推理执行 ──────────────────────────────────────────────────────────────────
inline bool trt_enqueue(nvinfer1::IExecutionContext * ctx,
                         void ** /*bindings*/,
                         cudaStream_t stream)
{
  return ctx->enqueueV3(stream);
}
