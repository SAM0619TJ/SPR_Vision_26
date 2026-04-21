#include "yolo26_trt.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "trt_compat.hpp"
#include "cuda_preprocess.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/path.hpp"

namespace auto_aim
{

namespace
{
auto runtime_deleter_v26 = [](nvinfer1::IRuntime * p) { if (p) delete p; };
auto engine_deleter_v26 = [](nvinfer1::ICudaEngine * p) { if (p) delete p; };
auto context_deleter_v26 = [](nvinfer1::IExecutionContext * p) { if (p) delete p; };

size_t get_dtype_size(nvinfer1::DataType dtype)
{
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF:  return sizeof(__half);
    default: return 0;
  }
}
}  // namespace

YOLO26_TRT::YOLO26_TRT(const std::string & config_path, bool debug)
: debug_(debug),
  runtime_(nullptr, runtime_deleter_v26),
  engine_(nullptr, engine_deleter_v26),
  context_(nullptr, context_deleter_v26),
  detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  {
    const auto raw_engine_path = yaml["yolo26_engine_path"].as<std::string>("");
    if (!raw_engine_path.empty()) {
      std::filesystem::path p(raw_engine_path);
      engine_path_ = p.is_absolute() ? p.string() : p.lexically_normal().string();
    }
  }
  onnx_path_ = tools::resolve_path_from_config(
    config_path, yaml["yolo26_onnx_path"].as<std::string>(""));
  const bool engine_exists = !engine_path_.empty() && std::filesystem::exists(engine_path_);
  const bool onnx_exists   = !onnx_path_.empty()   && std::filesystem::exists(onnx_path_);
  tools::logger()->info(
    "[YOLO26_TRT] model paths: engine='{}' (exists={}), onnx='{}' (exists={})",
    engine_path_, engine_exists, onnx_path_, onnx_exists);

  device_             = yaml["device"].as<std::string>("GPU");
  binary_threshold_   = yaml["threshold"].as<double>();
  min_confidence_     = yaml["min_confidence"].as<double>();
  score_threshold_    = yaml["yolo26_score_threshold"].as<float>(0.2f);
  nms_threshold_      = yaml["yolo26_nms_threshold"].as<float>(0.3f);
  swap_rb_channels_   = yaml["yolo26_trt_swap_rb"].as<bool>(false);
  use_cuda_preprocess_ = yaml["yolo26_use_cuda_preprocess"].as<bool>(false);

  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width  = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_    = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  cudaSetDevice(0);
  cudaStreamCreate(&stream_);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

  if (engine_exists) {
    tools::logger()->info("[YOLO26_TRT] Loading engine from: {}", engine_path_);
    loadEngine(engine_path_);
  } else if (onnx_exists) {
    tools::logger()->info("[YOLO26_TRT] Building engine from ONNX: {}", onnx_path_);
    buildEngineFromONNX(onnx_path_);
    if (!engine_path_.empty()) serializeEngine(engine_path_);
  } else {
    throw std::runtime_error(
      "No valid model found! Please provide either:\n"
      "  - yolo26_engine_path\n"
      "  - yolo26_onnx_path");
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("Failed to create execution context");

  // ── 获取 tensor 名称
  const char * input_name  = trt_get_tensor_name(engine_.get(), 0);
  const char * output_name = trt_get_tensor_name(engine_.get(), 1);

  // ── 获取数据类型
  input_dtype_  = trt_get_tensor_dtype(engine_.get(), input_name,  0);
  output_dtype_ = trt_get_tensor_dtype(engine_.get(), output_name, 1);

  // ── 获取静态输入形状
  auto input_dims = trt_get_tensor_shape(engine_.get(), input_name, 0);
  if (input_dims.nbDims == 4 && input_dims.d[2] > 0 && input_dims.d[3] > 0) {
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];
  }

  // 动态 shape 处理
  bool has_dynamic = false;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    if (input_dims.d[i] <= 0) { has_dynamic = true; break; }
  }
  if (has_dynamic) {
    nvinfer1::Dims4 fixed{1, 3, input_h_, input_w_};
    // ── 兼容层：TRT 10.x setInputShape，TRT 8.x setBindingDimensions
    if (!trt_set_input_shape(context_.get(), input_name, 0, fixed)) {
      throw std::runtime_error("Failed to set dynamic input shape for YOLO26_TRT");
    }
  }

  // ── 兼容层：获取运行时形状
  auto runtime_input_dims  = trt_get_context_tensor_shape(context_.get(), input_name,  0);
  auto runtime_output_dims = trt_get_context_tensor_shape(context_.get(), output_name, 1);

  if (runtime_input_dims.nbDims == 4 && runtime_input_dims.d[2] > 0 && runtime_input_dims.d[3] > 0) {
    input_h_ = runtime_input_dims.d[2];
    input_w_ = runtime_input_dims.d[3];
  }

  const size_t input_elements = static_cast<size_t>(1) * 3 * input_h_ * input_w_;
  input_size_bytes_ = input_elements * get_dtype_size(input_dtype_);

  size_t output_elements = 1;
  for (int i = 0; i < runtime_output_dims.nbDims; ++i) {
    output_elements *= static_cast<size_t>(
      runtime_output_dims.d[i] > 0 ? runtime_output_dims.d[i] : 1);
  }
  output_size_bytes_ = output_elements * get_dtype_size(output_dtype_);

  prepareOutputLayoutFromEngine(output_name);

  if (input_dtype_ == nvinfer1::DataType::kFLOAT) {
    input_host_float_.resize(input_elements);
  } else if (input_dtype_ == nvinfer1::DataType::kHALF) {
    input_host_half_.resize(input_elements);
  } else {
    throw std::runtime_error("YOLO26_TRT only supports FP32/FP16 input tensors");
  }

  if (output_dtype_ == nvinfer1::DataType::kFLOAT) {
    output_host_float_.resize(output_elements);
  } else if (output_dtype_ == nvinfer1::DataType::kHALF) {
    output_host_half_.resize(output_elements);
    output_host_float_.resize(output_elements);
  } else {
    throw std::runtime_error("YOLO26_TRT only supports FP32/FP16 output tensors");
  }

  cudaMalloc(reinterpret_cast<void **>(&buffers_[0]), input_size_bytes_);
  cudaMalloc(reinterpret_cast<void **>(&buffers_[1]), output_size_bytes_);

  // ── 兼容层：TRT 10.x 预绑定地址，TRT 8.x 为空操作
  trt_set_tensor_address(context_.get(), input_name,  0, buffers_[0]);
  trt_set_tensor_address(context_.get(), output_name, 1, buffers_[1]);

  input_image_ = cv::Mat(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));

  tools::logger()->info(
    "[YOLO26_TRT] Initialized: {}x{} input, output rows={}, stride={}, channel_first={}",
    input_w_, input_h_, output_rows_, output_stride_, output_channel_first_);
}

YOLO26_TRT::~YOLO26_TRT()
{
  if (gpu_img_buffer_) cudaFree(gpu_img_buffer_);
  if (buffers_[0]) cudaFree(buffers_[0]);
  if (buffers_[1]) cudaFree(buffers_[1]);
  if (stream_) cudaStreamDestroy(stream_);
}

void YOLO26_TRT::loadEngine(const std::string & engine_file)
{
  std::ifstream file(engine_file, std::ios::binary);
  if (!file.good()) throw std::runtime_error("Failed to open engine file: " + engine_file);

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) throw std::runtime_error("Failed to deserialize engine");
}

void YOLO26_TRT::buildEngineFromONNX(const std::string & onnx_file)
{
  auto builder = std::unique_ptr<nvinfer1::IBuilder, void(*)(nvinfer1::IBuilder*)>(
    nvinfer1::createInferBuilder(logger_), [](nvinfer1::IBuilder * p) { if (p) delete p; });

  const auto flags = 1U << static_cast<uint32_t>(
    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition, void(*)(nvinfer1::INetworkDefinition*)>(
    builder->createNetworkV2(flags), [](nvinfer1::INetworkDefinition * p) { if (p) delete p; });

  auto parser = std::unique_ptr<nvonnxparser::IParser, void(*)(nvonnxparser::IParser*)>(
    nvonnxparser::createParser(*network, logger_), [](nvonnxparser::IParser * p) { if (p) delete p; });

  if (!parser->parseFromFile(onnx_file.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX file: " + onnx_file);
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, void(*)(nvinfer1::IBuilderConfig*)>(
    builder->createBuilderConfig(), [](nvinfer1::IBuilderConfig * p) { if (p) delete p; });

  // ── 兼容层
  trt_set_workspace(config.get(), 1ULL << 30);

  // FP16 加速（Xavier NX Tensor Core 支持）
  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    tools::logger()->info("[YOLO26_TRT] FP16 enabled");
  }

  auto input_tensor = network->getInput(0);
  if (!input_tensor) throw std::runtime_error("Failed to get ONNX input tensor");

  auto input_dims = input_tensor->getDimensions();
  bool has_dynamic_input = false;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    if (input_dims.d[i] <= 0) { has_dynamic_input = true; break; }
  }

  if (has_dynamic_input) {
    auto profile = builder->createOptimizationProfile();
    if (!profile) throw std::runtime_error("Failed to create TensorRT optimization profile");

    nvinfer1::Dims min_dims = input_dims;
    nvinfer1::Dims opt_dims = input_dims;
    nvinfer1::Dims max_dims = input_dims;

    for (int i = 0; i < input_dims.nbDims; ++i) {
      if (input_dims.d[i] > 0) {
        min_dims.d[i] = opt_dims.d[i] = max_dims.d[i] = input_dims.d[i];
        continue;
      }
      if      (i == 0) { min_dims.d[i] = opt_dims.d[i] = max_dims.d[i] = 1; }
      else if (i == 1) { min_dims.d[i] = opt_dims.d[i] = max_dims.d[i] = 3; }
      else if (i == 2) { min_dims.d[i] = opt_dims.d[i] = max_dims.d[i] = input_h_; }
      else if (i == 3) { min_dims.d[i] = opt_dims.d[i] = max_dims.d[i] = input_w_; }
      else             { min_dims.d[i] = opt_dims.d[i] = max_dims.d[i] = 1; }
    }

    const char * input_name = input_tensor->getName();
    if (!profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, min_dims) ||
        !profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, opt_dims) ||
        !profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, max_dims)) {
      throw std::runtime_error("Failed to set TensorRT optimization profile dimensions");
    }
    config->addOptimizationProfile(profile);
  }

  auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)>(
    builder->buildSerializedNetwork(*network, *config),
    [](nvinfer1::IHostMemory * p) { if (p) delete p; });

  if (!serialized_engine) throw std::runtime_error("Failed to build TensorRT engine from ONNX");

  engine_.reset(runtime_->deserializeCudaEngine(
    serialized_engine->data(), serialized_engine->size()));
  if (!engine_) throw std::runtime_error("Failed to deserialize built engine");
}

void YOLO26_TRT::serializeEngine(const std::string & engine_file)
{
  auto serialized = std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)>(
    engine_->serialize(), [](nvinfer1::IHostMemory * p) { if (p) delete p; });

  if (!serialized) { tools::logger()->warn("[YOLO26_TRT] Failed to serialize engine"); return; }

  std::filesystem::create_directories(std::filesystem::path(engine_file).parent_path());
  std::ofstream file(engine_file, std::ios::binary);
  if (!file.good()) {
    tools::logger()->warn("[YOLO26_TRT] Failed to open {} for engine serialization", engine_file);
    return;
  }
  file.write(reinterpret_cast<const char *>(serialized->data()), serialized->size());
}

void YOLO26_TRT::prepareOutputLayoutFromEngine(const char * output_name)
{
  // ── 兼容层：获取运行时输出形状
  auto output_dims = trt_get_context_tensor_shape(context_.get(), output_name, 1);

  output_rows_ = 0;
  output_stride_ = 14;
  output_channel_first_ = false;

  if (output_dims.nbDims == 3) {
    if (output_dims.d[2] == 14 || output_dims.d[2] == 18 || output_dims.d[2] == 28) {
      output_rows_ = output_dims.d[1];
      output_stride_ = output_dims.d[2];
      output_channel_first_ = false;
      return;
    }
    if (output_dims.d[1] == 14 || output_dims.d[1] == 18 || output_dims.d[1] == 28) {
      output_rows_ = output_dims.d[2];
      output_stride_ = output_dims.d[1];
      output_channel_first_ = true;
      return;
    }
  }

  if (output_dims.nbDims == 2) {
    if (output_dims.d[1] == 14 || output_dims.d[1] == 18 || output_dims.d[1] == 28) {
      output_rows_ = output_dims.d[0];
      output_stride_ = output_dims.d[1];
      output_channel_first_ = false;
      return;
    }
    if (output_dims.d[0] == 14 || output_dims.d[0] == 18 || output_dims.d[0] == 28) {
      output_rows_ = output_dims.d[1];
      output_stride_ = output_dims.d[0];
      output_channel_first_ = true;
      return;
    }
  }

  throw std::runtime_error("YOLO26_TRT unsupported output shape");
}

std::list<Armor> YOLO26_TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) return {};

  cv::Mat bgr_img;
  tmp_img_ = raw_img;
  if (use_roi_) {
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  const int orig_w = bgr_img.cols;
  const int orig_h = bgr_img.rows;
  const float scale = std::min(
    static_cast<float>(input_w_) / orig_w, static_cast<float>(input_h_) / orig_h);
  const int new_w = static_cast<int>(orig_w * scale);
  const int new_h = static_cast<int>(orig_h * scale);
  const int pad_x = (input_w_ - new_w) / 2;
  const int pad_y = (input_h_ - new_h) / 2;

  input_image_.setTo(cv::Scalar(114, 114, 114));

  const bool use_cuda_path = use_cuda_preprocess_ && input_dtype_ == nvinfer1::DataType::kFLOAT;
  if (use_cuda_path) {
    const cv::Mat src_contiguous = bgr_img.isContinuous() ? bgr_img : bgr_img.clone();
    const size_t img_size =
      static_cast<size_t>(src_contiguous.cols) * src_contiguous.rows * 3 * sizeof(unsigned char);
    if (img_size > gpu_img_buffer_bytes_) {
      if (gpu_img_buffer_) { cudaFree(gpu_img_buffer_); gpu_img_buffer_ = nullptr; }
      cudaMalloc(reinterpret_cast<void **>(&gpu_img_buffer_), img_size);
      gpu_img_buffer_bytes_ = img_size;
    }
    cudaMemcpyAsync(gpu_img_buffer_, src_contiguous.data, img_size, cudaMemcpyHostToDevice, stream_);
    cuda_preprocess_letterbox(
      gpu_img_buffer_, src_contiguous.cols, src_contiguous.rows,
      reinterpret_cast<float *>(buffers_[0]), input_w_, input_h_, stream_);
  } else {
    cv::resize(
      bgr_img,
      input_image_(cv::Rect(pad_x, pad_y, new_w, new_h)),
      cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  }

  const size_t plane_size = static_cast<size_t>(input_h_) * input_w_;
  const int channel_map[3] = { swap_rb_channels_ ? 2 : 0, 1, swap_rb_channels_ ? 0 : 2 };

  if (use_cuda_path) {
    // CUDA 路径已直接写入 TRT 输入 buffer
  } else if (input_dtype_ == nvinfer1::DataType::kFLOAT) {
    for (int c = 0; c < 3; ++c)
      for (int h = 0; h < input_h_; ++h)
        for (int w = 0; w < input_w_; ++w)
          input_host_float_[c * plane_size + h * input_w_ + w] =
            input_image_.at<cv::Vec3b>(h, w)[channel_map[c]] / 255.0f;
    cudaMemcpyAsync(buffers_[0], input_host_float_.data(), input_size_bytes_,
                    cudaMemcpyHostToDevice, stream_);
  } else {
    for (int c = 0; c < 3; ++c)
      for (int h = 0; h < input_h_; ++h)
        for (int w = 0; w < input_w_; ++w)
          input_host_half_[c * plane_size + h * input_w_ + w] =
            __float2half(input_image_.at<cv::Vec3b>(h, w)[channel_map[c]] / 255.0f);
    cudaMemcpyAsync(buffers_[0], input_host_half_.data(), input_size_bytes_,
                    cudaMemcpyHostToDevice, stream_);
  }

  // ── 兼容层：TRT 10.x enqueueV3，TRT 8.x enqueueV2
  if (!trt_enqueue(context_.get(), buffers_, stream_)) {
    tools::logger()->error("[YOLO26_TRT] enqueue failed");
    return {};
  }

  if (output_dtype_ == nvinfer1::DataType::kFLOAT) {
    cudaMemcpyAsync(output_host_float_.data(), buffers_[1], output_size_bytes_,
                    cudaMemcpyDeviceToHost, stream_);
  } else {
    cudaMemcpyAsync(output_host_half_.data(), buffers_[1], output_size_bytes_,
                    cudaMemcpyDeviceToHost, stream_);
  }
  cudaStreamSynchronize(stream_);

  if (output_dtype_ == nvinfer1::DataType::kHALF) {
    for (size_t i = 0; i < output_host_half_.size(); ++i)
      output_host_float_[i] = __half2float(output_host_half_[i]);
  }

  const float * parse_ptr = output_host_float_.data();
  std::vector<float> transposed_output;
  if (output_channel_first_) {
    transposed_output.resize(static_cast<size_t>(output_rows_) * output_stride_);
    for (int r = 0; r < output_rows_; ++r)
      for (int c = 0; c < output_stride_; ++c)
        transposed_output[r * output_stride_ + c] = parse_ptr[c * output_rows_ + r];
    parse_ptr = transposed_output.data();
  }

  return parse(scale, pad_x, pad_y, parse_ptr, output_rows_, output_stride_,
               bgr_img, tmp_img_, frame_count);
}

std::list<Armor> YOLO26_TRT::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, 0, 0, output.ptr<float>(0), output.rows, output.cols,
               bgr_img, bgr_img, frame_count);
}

std::list<Armor> YOLO26_TRT::parse(
  float scale, int pad_x, int pad_y, const float * output_data, int num_rows, int stride,
  const cv::Mat & bgr_img, const cv::Mat & tmp_img, int frame_count)
{
  const int img_width  = bgr_img.cols;
  const int img_height = bgr_img.rows;
  const float inv_scale = 1.0f / scale;
  const float pad_x_f = static_cast<float>(pad_x);
  const float pad_y_f = static_cast<float>(pad_y);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  float max_conf_before_threshold = 0.0f;
  int max_conf_row = -1;

  for (int r = 0; r < num_rows; ++r) {
    const float * row_ptr = output_data + r * stride;

    float conf = 0.0f;
    int cls = 0;
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    int keypoint_start = 6;

    if (stride == 14 || stride == 18) {
      conf = row_ptr[4];
      cls  = static_cast<int>(row_ptr[5]);
      x1 = row_ptr[0]; y1 = row_ptr[1]; x2 = row_ptr[2]; y2 = row_ptr[3];
      keypoint_start = 6;
    } else if (stride >= 5 + class_num_ + 8) {
      const float obj_conf = row_ptr[4];
      float best_prob = 0.0f; int best_cls = 0;
      for (int c = 0; c < class_num_; ++c) {
        float p = row_ptr[5 + c];
        if (p > best_prob) { best_prob = p; best_cls = c; }
      }
      conf = obj_conf * best_prob;
      cls  = best_cls;
      x1 = row_ptr[0]; y1 = row_ptr[1]; x2 = row_ptr[2]; y2 = row_ptr[3];
      keypoint_start = 5 + class_num_;
    } else {
      continue;
    }

    if (conf > max_conf_before_threshold) { max_conf_before_threshold = conf; max_conf_row = r; }
    if (conf < score_threshold_) continue;

    const int box_x1 = std::clamp(static_cast<int>((x1 - pad_x_f) * inv_scale), 0, img_width);
    const int box_y1 = std::clamp(static_cast<int>((y1 - pad_y_f) * inv_scale), 0, img_height);
    const int box_x2 = std::clamp(static_cast<int>((x2 - pad_x_f) * inv_scale), 0, img_width);
    const int box_y2 = std::clamp(static_cast<int>((y2 - pad_y_f) * inv_scale), 0, img_height);
    const int w = box_x2 - box_x1;
    const int h = box_y2 - box_y1;
    if (w <= 0 || h <= 0) continue;

    std::vector<cv::Point2f> armor_key_points;
    armor_key_points.reserve(4);
    for (int i = 0; i < 4; ++i) {
      armor_key_points.emplace_back(
        (row_ptr[keypoint_start + i * 2]     - pad_x_f) * inv_scale,
        (row_ptr[keypoint_start + i * 2 + 1] - pad_y_f) * inv_scale);
    }

    ids.emplace_back(cls);
    confidences.emplace_back(conf);
    boxes.emplace_back(box_x1, box_y1, w, h);
    armors_key_points.emplace_back(std::move(armor_key_points));
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (int i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i],
                          offset_, YOLOVersion::YOLO26);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i],
                          YOLOVersion::YOLO26);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    const bool label_ok = (it->name != ArmorName::not_armor);
    const bool conf_ok  = (it->confidence > min_confidence_);
    const bool type_ok  = check_type(*it);
    if (!label_ok || !conf_ok || !type_ok) {
      it = armors.erase(it);
      continue;
    }
    it->center_norm = get_center_norm(tmp_img, it->center);
    ++it;
  }

  const bool should_log = (frame_count < 20) || (frame_count % 30 == 0) || armors.empty();
  if (should_log) {
    tools::logger()->info(
      "[YOLO26_TRT] frame={} rows={} nms_keep={} final={} max_conf={:.4f}@row{}",
      frame_count, num_rows, indices.size(), armors.size(),
      max_conf_before_threshold, max_conf_row);
  }

  if (debug_ && !tmp_img.empty()) draw_detections(tmp_img, armors, frame_count);

  return armors;
}

bool YOLO26_TRT::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}

bool YOLO26_TRT::check_type(const Armor & armor) const
{
  return (armor.type == ArmorType::small)
    ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
    : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
       armor.name != ArmorName::outpost);
}

cv::Point2f YOLO26_TRT::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return {center.x / bgr_img.cols, center.y / bgr_img.rows};
}

void YOLO26_TRT::sort_keypoints(std::vector<cv::Point2f> & keypoints) const
{
  if (keypoints.size() != 4) return;

  std::sort(keypoints.begin(), keypoints.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });

  std::vector<cv::Point2f> top    = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom = {keypoints[2], keypoints[3]};

  std::sort(top.begin(),    top.end(),    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });
  std::sort(bottom.begin(), bottom.end(), [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top[0];
  keypoints[1] = top[1];
  keypoints[2] = bottom[1];
  keypoints[3] = bottom[0];
}

void YOLO26_TRT::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  cv::Mat detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});

  for (const auto & armor : armors) {
    auto info = fmt::format("{:.2f} {} {} {}",
      armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
}

}  // namespace auto_aim
