#include "yolov5_trt.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <vector>

#include "trt_compat.hpp"
#include "tools/logger.hpp"
#include "tools/path.hpp"
#include "tasks/auto_aim/yolos/cuda_preprocess.hpp"

namespace auto_aim
{

auto runtime_deleter_v5 = [](nvinfer1::IRuntime * p) { if (p) delete p; };
auto engine_deleter_v5  = [](nvinfer1::ICudaEngine * p) { if (p) delete p; };
auto context_deleter_v5 = [](nvinfer1::IExecutionContext * p) { if (p) delete p; };

YOLOV5_TRT::YOLOV5_TRT(const std::string & config_path, bool debug)
: debug_(debug),
  detector_(config_path, false),
  runtime_(nullptr, runtime_deleter_v5),
  engine_(nullptr, engine_deleter_v5),
  context_(nullptr, context_deleter_v5)
{
  auto yaml = YAML::LoadFile(config_path);

  engine_path_ = tools::resolve_path_from_config(
    config_path, yaml["yolov5_engine_path"].as<std::string>(""));
  onnx_path_ = tools::resolve_path_from_config(
    config_path, yaml["yolov5_onnx_path"].as<std::string>(""));
  device_           = yaml["device"].as<std::string>("GPU");
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_   = yaml["min_confidence"].as<double>();
  use_traditional_  = yaml["use_traditional"].as<bool>();

  int x      = yaml["roi"]["x"].as<int>();
  int y      = yaml["roi"]["y"].as<int>();
  int width  = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_   = yaml["use_roi"].as<bool>();
  roi_       = cv::Rect(x, y, width, height);
  offset_    = cv::Point2f(x, y);

  cudaSetDevice(0);
  cudaStreamCreate(&stream_);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

  if (!engine_path_.empty() && std::filesystem::exists(engine_path_)) {
    tools::logger()->info("[YOLOV5_TRT] Loading engine from: {}", engine_path_);
    loadEngine(engine_path_);
  } else if (!onnx_path_.empty() && std::filesystem::exists(onnx_path_)) {
    tools::logger()->info("[YOLOV5_TRT] Building engine from ONNX: {}", onnx_path_);
    buildEngineFromONNX(onnx_path_);
    if (!engine_path_.empty()) serializeEngine(engine_path_);
  } else {
    throw std::runtime_error(
      "No valid model found! Provide yolov5_engine_path or yolov5_onnx_path in config.");
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("Failed to create execution context");

  const char * input_name  = trt_get_tensor_name(engine_.get(), 0);
  const char * output_name = trt_get_tensor_name(engine_.get(), 1);
  auto output_dims = trt_get_tensor_shape(engine_.get(), output_name, 1);

  // output shape: [1, num_detections, data_size]
  output_num_detections_ = output_dims.d[1];
  output_data_size_      = output_dims.d[2];

  tools::logger()->warn(
    "[YOLOV5_TRT] Output shape: [{},{},{}]",
    output_dims.d[0], output_num_detections_, output_data_size_);

  // input: [1, 1, 640, 640] grayscale float32
  input_size_  = 1 * 1 * input_h_ * input_w_ * sizeof(float);
  output_size_ = 1 * output_num_detections_ * output_data_size_ * sizeof(float);

  cudaMalloc(&buffers_[0], input_size_);
  cudaMalloc(&buffers_[1], output_size_);

  cudaMallocHost((void **)&output_host_, output_size_);

  gpu_img_buffer_size_ = 1920 * 1200 * 3 * sizeof(unsigned char);
  cudaMalloc(&gpu_img_buffer_, gpu_img_buffer_size_);

  trt_set_tensor_address(context_.get(), input_name,  0, buffers_[0]);
  trt_set_tensor_address(context_.get(), output_name, 1, buffers_[1]);

  tools::logger()->info(
    "[YOLOV5_TRT] Initialized: {}x{} grayscale input, {} detections x {} features",
    input_w_, input_h_, output_num_detections_, output_data_size_);
}

YOLOV5_TRT::~YOLOV5_TRT()
{
  cudaFreeHost(output_host_);
  cudaFree(gpu_img_buffer_);
  cudaFree(buffers_[0]);
  cudaFree(buffers_[1]);
  cudaStreamDestroy(stream_);
}

void YOLOV5_TRT::loadEngine(const std::string & engine_file)
{
  std::ifstream file(engine_file, std::ios::binary);
  if (!file.good()) throw std::runtime_error("Failed to open engine file: " + engine_file);

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);

  engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
  if (!engine_) throw std::runtime_error("Failed to deserialize engine");
}

void YOLOV5_TRT::buildEngineFromONNX(const std::string & onnx_file)
{
  auto builder = std::unique_ptr<nvinfer1::IBuilder, void(*)(nvinfer1::IBuilder*)>(
    nvinfer1::createInferBuilder(logger_), [](nvinfer1::IBuilder * p) { if (p) delete p; });

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition, void(*)(nvinfer1::INetworkDefinition*)>(
    builder->createNetworkV2(0U), [](nvinfer1::INetworkDefinition * p) { if (p) delete p; });

  auto parser = std::unique_ptr<nvonnxparser::IParser, void(*)(nvonnxparser::IParser*)>(
    nvonnxparser::createParser(*network, logger_), [](nvonnxparser::IParser * p) { if (p) delete p; });

  if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    throw std::runtime_error("Failed to parse ONNX file");

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, void(*)(nvinfer1::IBuilderConfig*)>(
    builder->createBuilderConfig(), [](nvinfer1::IBuilderConfig * p) { if (p) delete p; });

  trt_set_workspace(config.get(), 1U << 30);

  if (builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    tools::logger()->info("[YOLOV5_TRT] FP16 enabled");
  }

  // Handle dynamic batch dimension
  auto input = network->getInput(0);
  if (input && input->getDimensions().d[0] == -1) {
    auto profile = builder->createOptimizationProfile();
    // grayscale: [1, 1, 640, 640]
    nvinfer1::Dims4 dims{1, 1, input_h_, input_w_};
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);
  }

  auto serialized = std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)>(
    builder->buildSerializedNetwork(*network, *config), [](nvinfer1::IHostMemory * p) { if (p) delete p; });

  if (!serialized) throw std::runtime_error("Failed to build engine");

  engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
  if (!engine_) throw std::runtime_error("Failed to deserialize built engine");
}

void YOLOV5_TRT::serializeEngine(const std::string & engine_file)
{
  auto serialized = std::unique_ptr<nvinfer1::IHostMemory, void(*)(nvinfer1::IHostMemory*)>(
    engine_->serialize(), [](nvinfer1::IHostMemory * p) { if (p) delete p; });

  std::ofstream file(engine_file, std::ios::binary);
  if (!file.good()) {
    tools::logger()->error("[YOLOV5_TRT] Failed to open file for writing: {}", engine_file);
    return;
  }
  file.write(reinterpret_cast<const char *>(serialized->data()), serialized->size());
  tools::logger()->info("[YOLOV5_TRT] Engine serialized to: {}", engine_file);
}

std::list<Armor> YOLOV5_TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  cv::Mat bgr_img = use_roi_ ? raw_img(roi_).clone() : raw_img;

  float scale = std::min(
    static_cast<float>(input_w_) / bgr_img.cols,
    static_cast<float>(input_h_) / bgr_img.rows);

  // Upload BGR image to GPU, then CUDA kernel handles letterbox + BGR→Gray + normalize
  size_t img_size = bgr_img.cols * bgr_img.rows * 3 * sizeof(unsigned char);
  cudaMemcpyAsync(gpu_img_buffer_, bgr_img.data, img_size, cudaMemcpyHostToDevice, stream_);

  cuda_preprocess_letterbox_gray(
    gpu_img_buffer_,
    bgr_img.cols,
    bgr_img.rows,
    (float*)buffers_[0],
    input_w_,
    input_h_,
    stream_);

  trt_enqueue(context_.get(), buffers_, stream_);

  cudaMemcpyAsync(output_host_, buffers_[1], output_size_, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  return parse(scale, output_host_, output_num_detections_, bgr_img, frame_count);
}

std::list<Armor> YOLOV5_TRT::parse(
  double scale, float * output_data, int num_detections, const cv::Mat & bgr_img, int frame_count)
{
  // Wrap flat array as cv::Mat [num_detections, data_size] for convenient access
  cv::Mat output(num_detections, output_data_size_, CV_32F, output_data);

  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  for (int r = 0; r < output.rows; r++) {
    double score = sigmoid(output.at<float>(r, 8));
    if (score < score_threshold_) continue;

    cv::Mat color_scores   = output.row(r).colRange(9, 13);
    cv::Mat classes_scores = output.row(r).colRange(13, 22);
    cv::Point class_id, color_id;
    double score_num, score_color;
    cv::minMaxLoc(classes_scores, nullptr, &score_num, nullptr, &class_id);
    cv::minMaxLoc(color_scores,   nullptr, &score_color, nullptr, &color_id);

    // Keypoints: model outputs [TL, BL, BR, TR] (counterclockwise from top-left)
    // Rearrange to [TL, TR, BR, BL]
    std::vector<cv::Point2f> kpts = {
      {output.at<float>(r, 0) / static_cast<float>(scale), output.at<float>(r, 1) / static_cast<float>(scale)},  // TL
      {output.at<float>(r, 6) / static_cast<float>(scale), output.at<float>(r, 7) / static_cast<float>(scale)},  // TR
      {output.at<float>(r, 4) / static_cast<float>(scale), output.at<float>(r, 5) / static_cast<float>(scale)},  // BR
      {output.at<float>(r, 2) / static_cast<float>(scale), output.at<float>(r, 3) / static_cast<float>(scale)},  // BL
    };

    float min_x = kpts[0].x, max_x = kpts[0].x;
    float min_y = kpts[0].y, max_y = kpts[0].y;
    for (int i = 1; i < 4; i++) {
      min_x = std::min(min_x, kpts[i].x); max_x = std::max(max_x, kpts[i].x);
      min_y = std::min(min_y, kpts[i].y); max_y = std::max(max_y, kpts[i].y);
    }

    color_ids.emplace_back(color_id.x);
    num_ids.emplace_back(class_id.x);
    confidences.emplace_back(static_cast<float>(score));
    boxes.emplace_back(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y));
    armors_key_points.emplace_back(kpts);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (int idx : indices) {
    auto kpts = armors_key_points[idx];
    if (use_roi_) {
      for (auto & pt : kpts) pt += offset_;
      boxes[idx].x += static_cast<int>(offset_.x);
      boxes[idx].y += static_cast<int>(offset_.y);
      armors.emplace_back(color_ids[idx], num_ids[idx], confidences[idx], boxes[idx], kpts, offset_);
    } else {
      armors.emplace_back(color_ids[idx], num_ids[idx], confidences[idx], boxes[idx], kpts);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it) || !check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    if (use_traditional_) detector_.detect(*it, bgr_img);
    it->center_norm = {it->center.x / bgr_img.cols, it->center.y / bgr_img.rows};
    ++it;
  }

  return armors;
}

bool YOLOV5_TRT::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence >= min_confidence_;
}

bool YOLOV5_TRT::check_type(const Armor & armor) const
{
  return (armor.type == ArmorType::small)
    ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
    : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
       armor.name != ArmorName::outpost);
}

double YOLOV5_TRT::sigmoid(double x)
{
  return (x > 0) ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

std::list<Armor> YOLOV5_TRT::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output.ptr<float>(0), output.rows, bgr_img, frame_count);
}

}  // namespace auto_aim
