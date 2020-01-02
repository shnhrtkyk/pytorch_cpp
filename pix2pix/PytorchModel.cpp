#include <iostream>
#include "PytorchModel.h"

PytorchModel::PytorchModel()
{
	
	
}

/**
 * メッセージを出力する
 */
void PytorchModel::ShowMessage(const char* message)
{
	std::cout << "[message]" << message << std::endl;
}
void PytorchModel::LoadModel()
{
	module = torch::jit::load("./model.pt");
	torch::DeviceType device_type = torch::kCUDA;
	torch::Device device(device_type);
	module.to(device);
	
	output_tensor.to(device);
	input_tensor.to(device);
	original_tensor.to(device);
	predicted_tensor.to(device);
}
void PytorchModel::ReadImage(std::string path)
{
	// 入力画像を読み込む
	original_img = (cv::imread(path, 1));
	original_channel = original_img.channels();
	original_height = original_img.rows;
	original_width = original_img.cols;

	std::vector<int64_t> original_dims{ static_cast<int64_t>(1), // 1
						  static_cast<int64_t>(original_channel), // 3
						  static_cast<int64_t>(original_height), // h=512
						  static_cast<int64_t>(original_width) }; // w=512

	// 入力画像を読み込む
	cv::Mat mf_input = cv::Mat::zeros(original_height*original_channel, original_width, CV_32FC1);
	cv::Mat m_bgr[3], mf_rgb_rgb[3];
	cv::split(original_img, m_bgr);
	for (int i = 0; i < 3; i++) {
		m_bgr[i].convertTo(mf_rgb_rgb[2 - i], CV_32FC1, 1.0 / 128.0, -1);
		m_bgr[i].release();
	}

	// at::Tensorに変換予定のcv::Matに値をコピーしていく．
	for (int i = 0; i < 3; i++) {
		mf_rgb_rgb[i].copyTo(mf_input.rowRange(i*input_height, (i + 1)*input_height));
	}

	// 入力となるat::Tensorを生成
	at::TensorOptions options(at::kFloat);
	original_tensor = torch::from_blob(mf_input.data, at::IntList(original_dims), options);
	predicted_tensor = torch::from_blob(mf_input.data, at::IntList(original_dims), options);

	

}
std::vector<torch::jit::IValue>  PytorchModel::SetTensor(cv::Mat img)
{
	// 入力画像を読み込む
	cv::Mat mf_input = cv::Mat::zeros(input_height*input_channel, input_width, CV_32FC1);
	cv::Mat m_bgr[3], mf_rgb_rgb[3];
	cv::split(img, m_bgr);
	for (int i = 0; i < 3; i++) {
		m_bgr[i].convertTo(mf_rgb_rgb[2 - i], CV_32FC1, 1.0 / 128.0, -1);
		m_bgr[i].release();
	}
	   
	// at::Tensorに変換予定のcv::Matに値をコピーしていく．
	for (int i = 0; i < 3; i++) {
		mf_rgb_rgb[i].copyTo(mf_input.rowRange(i*input_height, (i + 1)*input_height));
	}

	// 入力となるat::Tensorを生成
	at::TensorOptions options(at::kFloat);
	input_tensor = torch::from_blob(mf_input.data, at::IntList(input_dims), options);
	//std::cout << input_tensor << std::endl;

	std::vector<torch::jit::IValue> input_img;
	torch::DeviceType device_type = torch::kCUDA;
	torch::Device device(device_type);
	input_img.push_back(input_tensor.to(device));
	return input_img;
}



void PytorchModel::Prediction(std::vector<torch::jit::IValue>  tensor)
{
	output_tensor = module.forward({ tensor }).toTensor();
}

cv::Mat PytorchModel::UnTanh()
{
	output_tensor_cpu = output_tensor.permute({ 0, 2, 3, 1 });

	output_tensor_cpu = output_tensor_cpu.squeeze(0).detach();
	output_tensor_cpu = output_tensor_cpu.mul(128).to(torch::kU8);
	output_tensor_cpu = output_tensor_cpu.add(128).to(torch::kU8);
	output_tensor_cpu = output_tensor_cpu.to(torch::kCPU);

	cv::Mat m_output(256, 256, CV_8UC3, output_tensor_cpu.data<unsigned char>());



	//std::cout << m_output.size() << std::endl;
	//std::cout << m_output.channels() << std::endl;
	//std::cout << m_output.type() << std::endl;
	//
	return m_output;
}