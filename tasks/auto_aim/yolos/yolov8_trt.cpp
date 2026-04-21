#include "yolov8_trt.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

#include "trt_compat.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/path.hpp"
#include "tasks/auto_aim/yolos/cuda_preprocess.hpp"

namespace auto_aim
{

auto runtime_deleter_v8 = [](nvinfer1::IRuntime* p) { if(p) delete p; };
auto engine_deleter_v8 = [](nvinfer1::ICudaEngine* p) { if(p) delete p; };
auto context_deleter_v8 = [](nvinfer1::IExecutionContext* p) { if(p) delete p; };

YOLOV8_TRT::YOLOV8_TRT(const std::string & config_path, bool debug)
: debug_(debug),
  detector_(config_path, false),
  runtime_(nullptr, runtime_deleter_v8),
  engine_(nullptr, engine_deleter_v8),
  context_(nullptr, context_deleter_v8)
{
  auto yaml = YAML::LoadFile(config_path);

  engine_path_ = tools::resolve_path_from_config(
    config_path, yaml["yolov8_engine_path"].as<std::string>(""));
  onnx_path_ = tools::resolve_path_from_config(
    config_path, yaml["yolov8_onnx_path"].as<std::string>(""));
  device_ = yaml["device"].as<std::string>("GPU");
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  use_async_inference_ = yaml["use_async_inference"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();

  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  cudaSetDevice(0);
  cudaStreamCreate(&stream_);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }

  if (!engine_path_.empty() && std::filesystem::exists(engine_path_)) {
    tools::logger()->info("[YOLOV8_TRT] Loading engine from: {}", engine_path_);
    loadEngine(engine_path_);
  } else if (!onnx_path_.empty() && std::filesystem::exists(onnx_path_)) {
    tools::logger()->info("[YOLOV8_TRT] Building engine from ONNX: {}", onnx_path_);
    buildEngineFromONNX(onnx_path_);
    if (!engine_path_.empty()) {
      serializeEngine(engine_path_);
    }
  } else {
    throw std::runtime_error(
      "No valid model found! Please provide either:\n"
      "  - yolov8_engine_path: path to .engine file\n"
      "  - yolov8_onnx_path: path to .onnx file");
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create execution context");
  }

  // ── 获取 tensor 名称和形状
  const char * input_name  = trt_get_tensor_name(engine_.get(), 0);
  const char * output_name = trt_get_tensor_name(engine_.get(), 1);
  auto input_dims  = trt_get_tensor_shape(engine_.get(), input_name,  0);
  auto output_dims = trt_get_tensor_shape(engine_.get(), output_name, 1);

  auto input_dtype  = trt_get_tensor_dtype(engine_.get(), input_name,  0);
  auto output_dtype = trt_get_tensor_dtype(engine_.get(), output_name, 1);

  std::string input_dtype_str  = (input_dtype  == nvinfer1::DataType::kFLOAT) ? "FP32" :
                                  (input_dtype  == nvinfer1::DataType::kHALF)  ? "FP16" : "OTHER";
  std::string output_dtype_str = (output_dtype == nvinfer1::DataType::kFLOAT) ? "FP32" :
                                  (output_dtype == nvinfer1::DataType::kHALF)  ? "FP16" : "OTHER";

  tools::logger()->info("[YOLOV8_TRT] ===== Engine Tensor Info =====");
  tools::logger()->info("[YOLOV8_TRT] Input: '{}' shape=[{},{},{},{}] dtype={}",
    input_name, input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3], input_dtype_str);
  tools::logger()->info("[YOLOV8_TRT] Output: '{}' shape=[{},{},{}] dtype={}",
    output_name, output_dims.d[0], output_dims.d[1], output_dims.d[2], output_dtype_str);
  tools::logger()->info("[YOLOV8_TRT] ==============================");

  input_w_ = input_dims.d[3];
  input_h_ = input_dims.d[2];

  // YOLOv8 输出格式 [1, num_features, num_detections]
  output_data_size_      = output_dims.d[1];
  output_num_detections_ = output_dims.d[2];

  input_size_  = 1 * 3 * input_h_ * input_w_ * sizeof(float);
  output_size_ = 1 * output_num_detections_ * output_data_size_ * sizeof(float);

  cudaMalloc(&buffers_[0], input_size_);
  cudaMalloc(&buffers_[1], output_size_);

  cudaError_t err_input = cudaMallocHost((void**)&input_host_, input_size_);
  if (err_input != cudaSuccess) {
    throw std::runtime_error("Failed to allocate input pinned memory");
  }

  size_t topk_buffer_size = output_data_size_ * topk_max_ * sizeof(float);
  cudaError_t err_topk = cudaMalloc(&gpu_topk_buffer_, topk_buffer_size);
  if (err_topk != cudaSuccess) {
    cudaFreeHost(input_host_);
    throw std::runtime_error("Failed to allocate GPU TopK buffer");
  }

  cudaError_t err_output = cudaMallocHost((void**)&output_host_, topk_max_ * output_data_size_ * sizeof(float));
  if (err_output != cudaSuccess) {
    cudaFreeHost(input_host_);
    cudaFree(gpu_topk_buffer_);
    throw std::runtime_error("Failed to allocate output pinned memory");
  }

  input_image_ = cv::Mat(input_h_, input_w_, CV_8UC3, cv::Scalar(0, 0, 0));

  gpu_img_buffer_size_ = 1920 * 1200 * 3 * sizeof(unsigned char);
  cudaError_t err_img = cudaMalloc(&gpu_img_buffer_, gpu_img_buffer_size_);
  if (err_img != cudaSuccess) {
    cudaFreeHost(input_host_);
    cudaFreeHost(output_host_);
    cudaFree(gpu_topk_buffer_);
    throw std::runtime_error("Failed to allocate GPU image buffer");
  }

  trt_set_tensor_address(context_.get(), input_name,  0, buffers_[0]);
  trt_set_tensor_address(context_.get(), output_name, 1, buffers_[1]);

  tools::logger()->info(
    "[YOLOV8_TRT] Initialized: {}x{} input, {} max detections, {} data per detection",
    input_w_, input_h_, output_num_detections_, output_data_size_);
}

YOLOV8_TRT::~YOLOV8_TRT()
{
  cudaFreeHost(input_host_);
  cudaFreeHost(output_host_);
  cudaFree(gpu_img_buffer_);
  cudaFree(gpu_topk_buffer_);
  cudaFree(buffers_[0]);
  cudaFree(buffers_[1]);
  cudaStreamDestroy(stream_);
}

void YOLOV8_TRT::loadEngine(const std::string & engine_file)
{
  std::ifstream file(engine_file, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("Failed to open engine file: " + engine_file);
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize engine");
  }

  tools::logger()->info("[YOLOV8_TRT] Engine loaded successfully");
}

void YOLOV8_TRT::buildEngineFromONNX(const std::string & onnx_file)
{
  auto builder = std::unique_ptr<nvinfer1::IBuilder, void(*)(nvinfer1::IBuilder*)>(
    nvinfer1::createInferBuilder(logger_), [](nvinfer1::IBuilder* p) { if(p) delete p; });

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition, void(*)(nvinfer1::INetworkDefinition*)>(
    builder->createNetworkV2(0U), [](nvinfer1::INetworkDefinition* p) { if(p) delete p; });

  auto parser = std::unique_ptr<nvonnxparser::IParser, void(*)(nvonnxparser::IParser*)>(
    nvonnxparser::createParser(*network, logger_), [](nvonnxparser::IParser* p) { if(p) delete p; });

  if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX file");
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, void(*)(nvinfer1::IBuilderConfig*)>(
    builder->createBuilderConfig(), [](nvinfer1::IBuilderConfig* p) { if(p) delete p; });

  trt_set_workspace(config.get(), 1U << 30);

  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    tools::logger()->info("[YOLOV8_TRT] FP16 enabled");
  }

  // 为动态shape输入添加optimization profile（固定batch=1, 640x640）
  auto input = network->getInput(0);
  if (input && input->getDimensions().d[0] == -1) {
    auto profile = builder->createOptimizationProfile();
    nvinfer1::Dims4 min_dims{1, 3, input_h_, input_w_};
    nvinfer1::Dims4 opt_dims{1, 3, input_h_, input_w_};
    nvinfer1::Dims4 max_dims{1, 3, input_h_, input_w_};
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    config->addOptimizationProfile(profile);
    tools::logger()->info("[YOLOV8_TRT] Added optimization profile for dynamic input");
  }

  auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)>(
    builder->buildSerializedNetwork(*network, *config), [](nvinfer1::IHostMemory* p) { if(p) delete p; });

  if (!serialized_engine) {
    throw std::runtime_error("Failed to build engine");
  }

  engine_.reset(runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize built engine");
  }

  tools::logger()->info("[YOLOV8_TRT] Engine built successfully");
}

void YOLOV8_TRT::serializeEngine(const std::string & engine_file)
{
  auto serialized = std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)>(
    engine_->serialize(), [](nvinfer1::IHostMemory* p) { if(p) delete p; });

  if (!serialized) {
    tools::logger()->error("[YOLOV8_TRT] Failed to serialize engine");
    return;
  }

  std::ofstream file(engine_file, std::ios::binary);
  if (!file.good()) {
    tools::logger()->error("[YOLOV8_TRT] Failed to open file for writing: {}", engine_file);
    return;
  }

  file.write(reinterpret_cast<const char *>(serialized->data()), serialized->size());
  file.close();

  tools::logger()->info("[YOLOV8_TRT] Engine serialized to: {}", engine_file);
}

std::list<Armor> YOLOV8_TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  auto t_start = std::chrono::high_resolution_clock::now();

  cv::Mat bgr_img;
  if (use_roi_) {
    bgr_img = raw_img(roi_).clone();
  } else {
    bgr_img = raw_img;
  }
  auto t_roi = std::chrono::high_resolution_clock::now();

  float scale = std::min(
    static_cast<float>(input_w_) / bgr_img.cols, static_cast<float>(input_h_) / bgr_img.rows);
  int pad_x = (input_w_ - static_cast<int>(bgr_img.cols * scale)) / 2;
  int pad_y = (input_h_ - static_cast<int>(bgr_img.rows * scale)) / 2;

  size_t img_size = bgr_img.cols * bgr_img.rows * 3 * sizeof(unsigned char);
  if (img_size > gpu_img_buffer_size_) {
    tools::logger()->error(
      "[YOLOV8_TRT] img_size {} exceeds gpu_img_buffer_size_ {}, img={}x{}",
      img_size, gpu_img_buffer_size_, bgr_img.cols, bgr_img.rows);
    cudaFree(gpu_img_buffer_);
    gpu_img_buffer_size_ = img_size;
    cudaMalloc(&gpu_img_buffer_, gpu_img_buffer_size_);
    tools::logger()->warn("[YOLOV8_TRT] GPU image buffer reallocated to {} bytes", gpu_img_buffer_size_);
  }
  cudaMemcpyAsync(gpu_img_buffer_, bgr_img.data, img_size, cudaMemcpyHostToDevice, stream_);

  cuda_preprocess_letterbox(
    gpu_img_buffer_,
    bgr_img.cols,
    bgr_img.rows,
    (float*)buffers_[0],
    input_w_,
    input_h_,
    stream_);

  auto t_preprocess = std::chrono::high_resolution_clock::now();

  if (use_async_inference_ && !first_frame_) {
    cudaStreamSynchronize(stream_);

    int actual_topk = cuda_topk_filter(
      (float*)buffers_[1],
      output_data_size_,
      output_num_detections_,
      topk_max_,
      gpu_topk_buffer_,
      topk_threshold_,
      stream_);

    size_t topk_size = output_data_size_ * actual_topk * sizeof(float);
    cudaMemcpyAsync(output_host_, gpu_topk_buffer_, topk_size, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // ── 推理
    trt_enqueue(context_.get(), buffers_, stream_);

    auto result = parse(prev_scale_, prev_pad_x_, prev_pad_y_, output_host_,
                       actual_topk, output_data_size_, prev_raw_img_, prev_frame_count_);

    auto t_clone_start = std::chrono::high_resolution_clock::now();
    prev_raw_img_ = bgr_img.clone();
    auto t_clone_end = std::chrono::high_resolution_clock::now();
    double clone_ms = std::chrono::duration<double, std::milli>(t_clone_end - t_clone_start).count();
    if (clone_ms > 2.0)
      tools::logger()->warn("[YOLOV8_TRT] async clone took {:.2f}ms ({}x{})",
        clone_ms, bgr_img.cols, bgr_img.rows);
    prev_scale_ = scale;
    prev_pad_x_ = pad_x;
    prev_pad_y_ = pad_y;
    prev_frame_count_ = frame_count;

    return result;
  } else {
    // ── 推理
    trt_enqueue(context_.get(), buffers_, stream_);
    auto t_infer = std::chrono::high_resolution_clock::now();

    int actual_topk = cuda_topk_filter(
      (float*)buffers_[1],
      output_data_size_,
      output_num_detections_,
      topk_max_,
      gpu_topk_buffer_,
      topk_threshold_,
      stream_);

    size_t topk_size = output_data_size_ * actual_topk * sizeof(float);
    cudaMemcpyAsync(output_host_, gpu_topk_buffer_, topk_size, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    auto t_d2h = std::chrono::high_resolution_clock::now();

    auto result = parse(scale, pad_x, pad_y, output_host_,
                       actual_topk, output_data_size_, bgr_img, frame_count);

    auto t_end = std::chrono::high_resolution_clock::now();
    tools::logger()->info(
      "[YOLOV8_TRT] Frame {}: total={:.2f}ms infer={:.2f}ms d2h={:.2f}ms",
      frame_count,
      std::chrono::duration<double, std::milli>(t_end - t_start).count(),
      std::chrono::duration<double, std::milli>(t_d2h - t_infer).count(),
      std::chrono::duration<double, std::milli>(t_end - t_d2h).count());

    if (use_async_inference_) {
      first_frame_ = false;
      auto t_clone_start2 = std::chrono::high_resolution_clock::now();
      prev_raw_img_ = bgr_img.clone();
      auto t_clone_end2 = std::chrono::high_resolution_clock::now();
      double clone_ms2 = std::chrono::duration<double, std::milli>(t_clone_end2 - t_clone_start2).count();
      if (clone_ms2 > 2.0)
        tools::logger()->warn("[YOLOV8_TRT] sync clone took {:.2f}ms ({}x{})",
          clone_ms2, bgr_img.cols, bgr_img.rows);
      prev_scale_ = scale;
      prev_pad_x_ = pad_x;
      prev_pad_y_ = pad_y;
      prev_frame_count_ = frame_count;
    }

    return result;
  }
}

std::list<Armor> YOLOV8_TRT::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  float* data = output.ptr<float>(0);
  return parse(scale, 0, 0, data, output.cols, output.rows, bgr_img, frame_count);
}

std::list<Armor> YOLOV8_TRT::parse(
  float scale, int pad_x, int pad_y, float* output_data, int num_detections, int data_size,
  const cv::Mat & bgr_img, int frame_count)
{
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  std::vector<std::vector<cv::Point2f>> keypoints_list;

  const float inv_scale = 1.0f / scale;

  for (int i = 0; i < num_detections; ++i) {
    float max_prob = 0.0f;
    int class_id = 0;
    for (int c = 0; c < 16; ++c) {
      float prob = output_data[(4 + c) * num_detections + i];
      if (prob > max_prob) {
        max_prob = prob;
        class_id = c;
      }
    }
    float conf = max_prob;

    if (conf < score_threshold_) continue;

    float cx = output_data[0 * num_detections + i];
    float cy = output_data[1 * num_detections + i];
    float w  = output_data[2 * num_detections + i];
    float h  = output_data[3 * num_detections + i];

    float x1 = (cx - w / 2 - pad_x) * inv_scale;
    float y1 = (cy - h / 2 - pad_y) * inv_scale;
    float x2 = (cx + w / 2 - pad_x) * inv_scale;
    float y2 = (cy + h / 2 - pad_y) * inv_scale;

    cv::Rect box(
      static_cast<int>(x1), static_cast<int>(y1),
      static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));

    std::vector<cv::Point2f> kpts_raw;
    for (int k = 0; k < 4; ++k) {
      float kx = (output_data[(20 + k * 2) * num_detections + i] - pad_x) * inv_scale;
      float ky = (output_data[(20 + k * 2 + 1) * num_detections + i] - pad_y) * inv_scale;
      kpts_raw.emplace_back(kx, ky);
    }

    // 模型输出 [TR, BR, BL, TL] → 重排为 [TL, TR, BR, BL]
    std::vector<cv::Point2f> kpts = {kpts_raw[3], kpts_raw[0], kpts_raw[1], kpts_raw[2]};

    boxes.push_back(box);
    confidences.push_back(conf);
    class_ids.push_back(class_id);
    keypoints_list.push_back(kpts);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (int idx : indices) {
    auto & kpts = keypoints_list[idx];

    if (use_roi_) {
      for (auto & pt : kpts) pt += offset_;
      boxes[idx].x += static_cast<int>(offset_.x);
      boxes[idx].y += static_cast<int>(offset_.y);
    }

    try {
      Armor armor(class_ids[idx], confidences[idx], boxes[idx], kpts, YOLOVersion::YOLOV8_TRT);

      if (!check_name(armor) || !check_type(armor)) continue;

      armors.push_back(armor);
    } catch (const std::exception& e) {
      tools::logger()->warn("[YOLOV8_TRT] Failed to create armor: {}", e.what());
      continue;
    }
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV8_TRT::check_name(const Armor & armor) const
{
  return armor.confidence >= min_confidence_;
}

bool YOLOV8_TRT::check_type(const Armor & armor) const
{
  if (armor.points.size() != 4) return false;
  float width  = cv::norm(armor.points[1] - armor.points[0]);
  float height = cv::norm(armor.points[2] - armor.points[1]);
  float ratio  = width / std::max(height, 1.0f);
  return ratio > 0.5f && ratio < 5.0f;
}

void YOLOV8_TRT::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  cv::Mat vis = img.clone();
  for (const auto & armor : armors) {
    for (size_t i = 0; i < armor.points.size(); ++i) {
      cv::circle(vis, armor.points[i], 3, cv::Scalar(0, 255, 0), -1);
      cv::line(vis, armor.points[i], armor.points[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
    }
    cv::putText(vis, fmt::format("{:.2f}", armor.confidence), armor.points[0],
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
  }
  cv::resize(vis, vis, {}, 0.5, 0.5);
  cv::imshow("YOLOV8_TRT Detection", vis);
  cv::waitKey(1);
}

}  // namespace auto_aim
