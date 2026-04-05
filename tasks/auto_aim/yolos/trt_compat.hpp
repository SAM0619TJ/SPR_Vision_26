#pragma once
// trt_compat.hpp — TensorRT 8.x / 10.x API 兼容层
// Xavier NX (JetPack 5.x) 搭载 TRT 8.5，Orin NX 搭载 TRT 10.x
// 通过宏统一两套 API，三个 yolo_trt 文件只需 include 此头文件

#include <NvInfer.h>

#define TRT_VERSION_INT (NV_TENSORRT_MAJOR * 10000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH)

// ── 内存池 / Workspace ────────────────────────────────────────────────────────
// TRT 10.x: setMemoryPoolLimit(kWORKSPACE, size)
// TRT 8.x : setMaxWorkspaceSize(size)
inline void trt_set_workspace(nvinfer1::IBuilderConfig * cfg, size_t size)
{
#if TRT_VERSION_INT >= 100000
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size);
#else
  cfg->setMaxWorkspaceSize(size);
#endif
}

// ── Tensor 名称查询 ───────────────────────────────────────────────────────────
// TRT 10.x: getIOTensorName(index)
// TRT 8.x : getBindingName(index)
inline const char * trt_get_tensor_name(nvinfer1::ICudaEngine * engine, int index)
{
#if TRT_VERSION_INT >= 100000
  return engine->getIOTensorName(index);
#else
  return engine->getBindingName(index);
#endif
}

// ── Tensor 形状查询（engine 静态形状）────────────────────────────────────────
// TRT 10.x: getTensorShape(name)
// TRT 8.x : getBindingDimensions(index)
inline nvinfer1::Dims trt_get_tensor_shape(nvinfer1::ICudaEngine * engine,
                                            const char * name, int index)
{
#if TRT_VERSION_INT >= 100000
  return engine->getTensorShape(name);
#else
  (void)name;
  return engine->getBindingDimensions(index);
#endif
}

// ── Tensor 形状查询（context 运行时形状）─────────────────────────────────────
// TRT 10.x: context->getTensorShape(name)
// TRT 8.x : context->getBindingDimensions(index)
inline nvinfer1::Dims trt_get_context_tensor_shape(nvinfer1::IExecutionContext * ctx,
                                                    const char * name, int index)
{
#if TRT_VERSION_INT >= 100000
  return ctx->getTensorShape(name);
#else
  (void)name;
  return ctx->getBindingDimensions(index);
#endif
}

// ── 动态输入形状设置 ──────────────────────────────────────────────────────────
// TRT 10.x: context->setInputShape(name, dims)
// TRT 8.x : context->setBindingDimensions(index, dims)
inline bool trt_set_input_shape(nvinfer1::IExecutionContext * ctx,
                                 const char * name, int index,
                                 const nvinfer1::Dims & dims)
{
#if TRT_VERSION_INT >= 100000
  return ctx->setInputShape(name, dims);
#else
  (void)name;
  return ctx->setBindingDimensions(index, dims);
#endif
}

// ── Tensor 数据类型查询 ───────────────────────────────────────────────────────
// TRT 10.x: getTensorDataType(name)
// TRT 8.x : getBindingDataType(index)
inline nvinfer1::DataType trt_get_tensor_dtype(nvinfer1::ICudaEngine * engine,
                                                const char * name, int index)
{
#if TRT_VERSION_INT >= 100000
  return engine->getTensorDataType(name);
#else
  (void)name;
  return engine->getBindingDataType(index);
#endif
}

// ── Buffer 地址绑定 + 推理执行 ────────────────────────────────────────────────
// TRT 10.x: setTensorAddress(name, ptr) + enqueueV3(stream)
// TRT 8.x : enqueueV2(bindings[], stream, nullptr)
//
// 使用方式：
//   trt_set_tensor_address(ctx, input_name, 0, input_buf);
//   trt_set_tensor_address(ctx, output_name, 1, output_buf);
//   trt_enqueue(ctx, {input_buf, output_buf}, stream);
inline void trt_set_tensor_address(nvinfer1::IExecutionContext * ctx,
                                    const char * name, int /*index*/,
                                    void * ptr)
{
#if TRT_VERSION_INT >= 100000
  ctx->setTensorAddress(name, ptr);
#else
  // TRT 8.x 不需要预先绑定地址，在 enqueueV2 时传入 bindings 数组
  (void)ctx; (void)name; (void)ptr;
#endif
}

// 推理执行：TRT 8.x 需要 bindings 数组，TRT 10.x 已通过 setTensorAddress 绑定
// bindings 参数在 TRT 10.x 下被忽略
inline bool trt_enqueue(nvinfer1::IExecutionContext * ctx,
                         void ** bindings,   // [input_buf, output_buf]
                         cudaStream_t stream)
{
#if TRT_VERSION_INT >= 100000
  (void)bindings;
  return ctx->enqueueV3(stream);
#else
  ctx->enqueueV2(bindings, stream, nullptr);
  return true;
#endif
}
