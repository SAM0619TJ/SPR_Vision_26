#ifndef AUTO_AIM__YOLOV5_TRT_HPP
#define AUTO_AIM__YOLOV5_TRT_HPP

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

class YOLOV5_TRT : public YOLOBase
{
public:
  YOLOV5_TRT(const std::string & config_path, bool debug);
  ~YOLOV5_TRT();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_;
  std::string engine_path_;
  std::string onnx_path_;
  bool debug_, use_roi_, use_traditional_;

  const float nms_threshold_ = 0.3f;
  const float score_threshold_ = 0.7f;
  double min_confidence_, binary_threshold_;

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_;

  cudaStream_t stream_;
  void * buffers_[2];
  float * output_host_;
  unsigned char * gpu_img_buffer_;
  size_t gpu_img_buffer_size_;

  size_t input_size_;
  size_t output_size_;

  const int input_w_ = 640;
  const int input_h_ = 640;
  // YOLOv5 anchor-based: 25200 detections, 22 features each
  // features: [kpt0x,kpt0y,kpt1x,kpt1y,kpt2x,kpt2y,kpt3x,kpt3y, obj, c0..c3(color), n0..n8(num)]
  int output_num_detections_ = 25200;
  int output_data_size_ = 22;

  cv::Rect roi_;
  cv::Point2f offset_;
  Detector detector_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  std::list<Armor> parse(
    double scale, float * output_data, int num_detections, const cv::Mat & bgr_img,
    int frame_count);
  double sigmoid(double x);

  void loadEngine(const std::string & engine_file);
  void buildEngineFromONNX(const std::string & onnx_file);
  void serializeEngine(const std::string & engine_file);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_TRT_HPP
