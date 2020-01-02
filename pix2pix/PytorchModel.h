#pragma once
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <ATen/ATen.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>


class PytorchModel
{
public:

	PytorchModel();

	void ShowMessage(const char* message);
	void LoadModel();
	void ReadImage(std::string path);
	std::vector<torch::jit::IValue>  SetTensor(cv::Mat img);
	void Prediction(std::vector<torch::jit::IValue>  tensor);
	cv::Mat UnTanh();


	torch::jit::script::Module module;

	cv::Mat original_img; // ì«Ç›çûÇÒÇæâÊëú
	int original_channel;// = original_img.channels();
	int original_height;// = original_img.rows;
	int original_width;// = original_img.cols;
	
	at::Tensor original_tensor = torch::ones({ 1, 3, 256, 256 });
	at::Tensor predicted_tensor = torch::ones({ 1, 3, 256, 256 });

	cv::Mat input_img;
	int input_channel = 3;// = original_img.channels();
	int input_height = 256;// = original_img.rows;
	int input_width = 256;// = original_img.cols;
	std::vector<int64_t> input_dims{ static_cast<int64_t>(1), // 1
						  static_cast<int64_t>(input_channel), // 3
						  static_cast<int64_t>(input_height), // h=512
						  static_cast<int64_t>(input_width) }; // w=512

	at::Tensor input_tensor = torch::ones({ 1, input_channel, input_height, input_width });
	at::Tensor output_tensor = torch::ones({ 1, input_channel, input_height, input_width });
	at::Tensor output_tensor_cpu = torch::ones({ 1, input_channel, input_height, input_width });
	
	// Parameters of your slideing window

	int windows_n_rows = 256;
	int windows_n_cols = 256;
	// Step of each window
	int StepSlide = 256;
	
	
};



	

