#ifndef AUTO_AIM__YOLOV8_TRT_HPP
#define AUTO_AIM__YOLOV8_TRT_HPP

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/yolos/tensorrt_logger.hpp"

namespace auto_aim
{

class YOLOV8_TRT : public YOLOBase
{
public:
  YOLOV8_TRT(const std::string & config_path, bool debug);
  ~YOLOV8_TRT();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_;
  std::string engine_path_;
  std::string onnx_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;
  bool use_async_inference_;
  bool use_traditional_;

  const int class_num_ = 16;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_;

  cudaStream_t stream_;

  void * buffers_[2];
  cv::Mat input_image_;
  float * input_host_;
  float * output_host_;

  unsigned char* gpu_img_buffer_;
  size_t gpu_img_buffer_size_;

  float* gpu_topk_buffer_;
  int topk_max_ = 1000;
  float topk_threshold_ = 0.05f;

  size_t input_size_;
  size_t output_size_;

  int input_w_ = 416;
  int input_h_ = 416;
  int output_num_detections_;
  int output_data_size_;

  bool first_frame_ = true;
  cv::Mat prev_raw_img_;
  float prev_scale_ = 1.0f;
  int prev_pad_x_ = 0;
  int prev_pad_y_ = 0;
  int prev_frame_count_ = 0;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;
  Detector detector_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  std::list<Armor> parse(
    float scale, int pad_x, int pad_y, float* output_data, int num_detections, int data_size,
    const cv::Mat & bgr_img, int frame_count);
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;

  void loadEngine(const std::string & engine_file);
  void buildEngineFromONNX(const std::string & onnx_file);
  void serializeEngine(const std::string & engine_file);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV8_TRT_HPP
