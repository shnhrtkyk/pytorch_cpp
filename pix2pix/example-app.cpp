#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <ATen/ATen.h>
#include "PytorchModel.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

//at::Tensor  maketensor(cv::Mat pm_images) {
//	// Moduleに入力するat::Tensorの次元を定義
//	int channel = pm_images.channels();
//	const int height = pm_images.rows;
//	const int width = pm_images.cols;
//	std::vector<int64_t> dims{ static_cast<int64_t>(1), // 1
//							  static_cast<int64_t>(channel), // 6
//							  static_cast<int64_t>(height), // h=512
//							  static_cast<int64_t>(width) }; // w=512
//
//	// 入力Tensorに変換する予定のcv::Matを作っておく．
//	// cv::Matのサイズも入力のat::Tensorに合わせておく
//	cv::Mat mf_input = cv::Mat::zeros(height*channel , width, CV_32FC1);
//
//	// OpenCVではRGBではなくBGRの順に値が入っているので，入れ替える
//	// DepthNet本家がやっているような下準備もここで行う．
//	cv::Mat m_bgr[3], mf_rgb_rgb[3];
//	cv::split(pm_images, m_bgr);
//	for (int i = 0; i < 3; i++) {
//		m_bgr[i].convertTo(mf_rgb_rgb[2 - i], CV_32FC1, 1.0 / 128.0, -1);
//		m_bgr[i].release();
//	}
//
//
//
//	// at::Tensorに変換予定のcv::Matに値をコピーしていく．
//	for (int i = 0; i < 3; i++) {
//		mf_rgb_rgb[i].copyTo(mf_input.rowRange(i*height, (i + 1)*height));
//	}
//
//
//	// 入力となるat::Tensorを生成
//	at::TensorOptions options(at::kFloat);
//	at::Tensor input_tensor = torch::from_blob(mf_input.data, at::IntList(dims), options);
//	//std::cout << input_tensor << std::endl;
//	return input_tensor;
//
//}
//
//void print_info(const cv::Mat& mat)
//{
//	using namespace std;
//
//	// 要素の型とチャンネル数の組み合わせ。
//	// 紙面の都合により、サンプルで使用する値のみ記述
//	cout << "type: " << (
//		mat.type() == CV_8UC3 ? "CV_8UC3" :
//		mat.type() == CV_16SC1 ? "CV_16SC1" :
//		mat.type() == CV_64FC2 ? "CV_64FC2" :
//		mat.type() == CV_64FC3 ? "CV_64FC3" :
//		"other"
//		) << endl;
//
//	// 要素の型
//	cout << "depth: " << (
//		mat.depth() == CV_8U ? "CV_8U" :
//		mat.depth() == CV_16S ? "CV_16S" :
//		mat.depth() == CV_64F ? "CV_64F" :
//		"other"
//		) << endl;
//
//	// チャンネル数
//	cout << "channels: " << mat.channels() << endl;
//
//	// バイト列が連続しているか
//	cout << "continuous: " <<
//		(mat.isContinuous() ? "true" : "false") << endl;
//}
//
//int getminmax(cv::Mat in) {
//
//	//最大と最小の初期値を設定（minが適当なのはやばそう）
//	double max = 0;
//	double min = 9999;
//
//	//めんどくさいから入力画像をclone
//	cv::Mat out = in.clone();
//
//	//最大と最小を取得
//	for (int y = 0; y < in.rows; ++y) {
//		for (int x = 0; x < in.cols; ++x) {
//			// 画像のチャネル数分だけループ。白黒の場合は1回、カラーの場合は3回　　　　　
//			for (int c = 0; c < in.channels(); ++c) {
//				float intensity = in.data[y * in.step + x * in.elemSize() + c];
//
//				//std::cout << intensity << std::endl;
//
//				if (max < intensity ) {
//					max = intensity;
//				}
//				if (min > intensity) {
//					min = intensity;
//				}
//
//			}
//		}
//	}
//
//	std::cout << "min = " << min << " , max = " << max<< "\n";
//
//
//
//	return 0;
//}
//at::Tensor tanh(cv::Mat in, std::vector<int64_t> dims) {
//
//	//めんどくさいから入力画像をclone
//	cv::Mat out; //= in.clone();
//	in.convertTo(out, CV_32FC3, 1.0/255);
//
//
//
//	cv::Mat image = in;
//	std::vector<int64_t> sizes = { 1, 3, out.rows, out.cols };
//	at::TensorOptions options(at::kFloat);
//
//	at::Tensor input_tensor = torch::from_blob(in.data, at::IntList(dims), options);
//	input_tensor = (input_tensor / 128.0) - 1.0;
//	input_tensor = input_tensor.toType(at::kFloat);
//	//std::cout <<  input_tensor << std::endl;
//	//cv::Mat mat(256, 256, CV_32FC3, tensor_image. template data<float>());
//
//
//	//-1~1になったMatを返す
//	return input_tensor;
//}
//at::Tensor  untanh(at::Tensor in) {
//	
//	at::TensorOptions options(at::kFloat);
//	std::cout << in << std::endl;
//	in = (in * 128.0) + 128.0;
//	in = in.toType(at::kFloat);
//	std::cout << in << std::endl;
//	//cv::Mat mat(256, 256, CV_32FC3, tensor_image. template data<float>());
//
//
//
//	//0~255になったMatを返す
//	return in;
//}
///* 画像の表示のサンプルコード C++版 */
//int sample_DisplayImage_Cpp(cv::Mat showimg) {
//	
//
//
//	/* ウィンドウの作成 */
//	cv::namedWindow("lenna", CV_WINDOW_AUTOSIZE);
//	/* 画像の表示 */
//	cv::imshow("lenna", showimg);
//
//	/* キー入力待ち */
//	cv::waitKey(0);
//
//
//
//	return 0;
//}
//void tensor2img(at::Tensor output_tensor) {
//	// 出力のat::Tensorをcv::Matへ格納
//	cv::Mat m_output(cv::Size(output_tensor.size(2)/*128*/, output_tensor.size(3)/*128*/), CV_32FC3, output_tensor.data<float>());
//	cv::Mat m_depth = m_output.clone();
//	cv::Mat m_upscaled_depth;
//	cv::resize(m_depth, m_upscaled_depth, cv::Size(output_tensor.size(1)/*512*/, output_tensor.size(2)/*512*/), 0, 0);
//	const float max_value = 255; // DepthNet本家が決めてた
//	cv::Mat m_arranged_depth = (128.0*m_upscaled_depth) + 128.0;
//	for (int i = 0; i < m_arranged_depth.rows; i++) {
//		for (int j = 0; j < m_arranged_depth.cols; j++) {
//			if (m_arranged_depth.at<float>(i, j) > 255.0) {
//				m_arranged_depth.at<float>(i, j) = 255.0;
//			}
//			else if (m_arranged_depth.at<float>(i, j) < 0.0) {
//				m_arranged_depth.at<float>(i, j) = 0.0;
//			}
//		}
//	}
//	//m_arranged_depth.convertTo(m_arranged_depth, CV_8U);
//	//cv::Mat m_color_map;
//	//cv::applyColorMap(m_arranged_depth, m_color_map, cv::COLORMAP_RAINBOW);
//}
//
//std::vector<torch::jit::IValue> imgread(std::string path) {
//	// 入力画像をペアで読み込む
//	auto pm_images =(cv::imread(path, 1));
//	//sample_DisplayImage_Cpp(pm_images);
//	//getminmax(pm_images);
//	//pm_images = tanh(pm_images);
//	//getminmax(pm_images);
//	// Moduleに入力するat::Tensorの次元を定義
//	const int channel = pm_images.channels();
//	const int height = pm_images.rows;
//	const int width = pm_images.cols;
//	std::vector<int64_t> dims{ static_cast<int64_t>(1), // 1
//							  static_cast<int64_t>(channel), // 3
//							  static_cast<int64_t>(height), // h=512
//							  static_cast<int64_t>(width) }; // w=512
//
//
//	// BGR to RGB which is what our network was trained on
//	//cv::cvtColor(pm_images, pm_images, cv::COLOR_BGR2RGB);
//	getminmax(pm_images);
//	// Resizing while preserving aspect ratio, comment out to run
//	// it on the whole input image.
//
//	at::Tensor input_tensor = maketensor(pm_images);
//
//	// 入力となるat::Tensorを生成
//	at::TensorOptions options(at::kFloat);
//	std::vector<torch::jit::IValue> input_img;
//	torch::DeviceType device_type = torch::kCUDA;
//	torch::Device device(device_type);
//	//input_img.push_back(torch::ones({ 1, 3, 256, 256 }).to(device));
//	input_img.push_back((input_tensor).to(device));
//
//
//	std::cout << "input_img dim = [" << input_tensor.size(0)
//		<< " x " << input_tensor.size(1)
//		<< " x " << input_tensor.size(2)
//		<< " x " << input_tensor.size(3)
//		<< "]\n";
//	return(input_img);
//}
int main(int argc, const char* argv[]){
	PytorchModel pm;
	pm.ShowMessage("Hello!");
	try {
		pm.LoadModel();
	}
	catch (const c10::Error& e) {
		pm.ShowMessage("Model Loading error!");
		std::cerr << e.what();

	}
	pm.ShowMessage("Model Loading passed!");


	/* ReadImage And Sliding Window */
	cv::Mat DrawResultGrid =  (cv::imread("./test.png", 1));
	std::cout << DrawResultGrid.size() << std::endl;
	std::cout << DrawResultGrid.channels() << std::endl;
	std::cout << DrawResultGrid.type() << std::endl;
	// Cycle row step
	for (int row = 0; row <= DrawResultGrid.rows - pm.windows_n_rows; row += pm.StepSlide)
	{
		// Cycle col step
		for (int col = 0; col <= DrawResultGrid.cols - pm.windows_n_cols; col += pm.StepSlide)
		{
			// There could be feature evaluator  over Windows

			// resulting window   
			cv::Rect windows(col, row, pm.windows_n_rows, pm.windows_n_cols);


			cv::Mat Roi = DrawResultGrid(windows);

			std::vector<torch::jit::IValue>  tmp_imput;
			tmp_imput = pm.SetTensor(Roi);
			pm.Prediction(tmp_imput);
			cv::Mat tmp_output = pm.UnTanh();
			tmp_output.copyTo(DrawResultGrid.rowRange(col, col + pm.windows_n_rows).colRange(row , row + pm.windows_n_cols));
			
		}
	}
	cv::imwrite("./all.jpg", DrawResultGrid);
	try {
		pm.ReadImage("./00000016.png");
	}
	catch (const c10::Error& e) {
		pm.ShowMessage("Image Reading error!");
		std::cerr << e.what();

	}
	pm.ShowMessage("Image Reading passed!");

	//try {
	//	//std::cout << pm.input_tensor << std::endl;
	//	pm.PredictionForOriginalImage();
	//	//std::cout << pm.output_tensor << std::endl;
	//}
	//catch (const c10::Error& e) {
	//	pm.ShowMessage("PredictionForOriginalImage error!");
	//	std::cerr << e.what();

	//}
	//pm.ShowMessage("PredictionForOriginalImage passed!");

	std::vector<torch::jit::IValue>  input;
	try {
		input = pm.SetTensor(pm.original_img);
	}
	catch (const c10::Error& e) {
		pm.ShowMessage("Set Tensor error!");
		std::cerr << e.what();

	}
	pm.ShowMessage("Set Tensor passed!");



	try {
		//std::cout << pm.input_tensor << std::endl;
		pm.Prediction(input);
		//std::cout << pm.output_tensor << std::endl;
	}
	catch (const c10::Error& e) {
		pm.ShowMessage("Prediction error!");
		std::cerr << e.what();

	}
	pm.ShowMessage("Prediction passed!");	


	try {
		cv::Mat m_output = pm.UnTanh();
		cv::imwrite("./pred.jpg", m_output);
	}
	catch (const c10::Error& e) {
		pm.ShowMessage("UnTanh error!");
		std::cerr << e.what();

	}
	pm.ShowMessage("UnTanh passed!");

}


//int main(int argc, const char* argv[]) {
//	if (argc != 1) {
//		std::cerr << "arg1: example-app <path-to-exported-script-module>\n";
//		std::cerr << "arg2: example-app <path-to-input image>\n";
//		return -1;
//	}
//
//	PytorchModel pm;
//	pm.ShowMessage("Hello!");
//
//	torch::jit::script::Module module;
//
//	try {
//		// Deserialize the ScriptModule from a file using torch::jit::load().
//		//module = torch::jit::load(argv[1]);
//		module = torch::jit::load("./model.pt");
//	
//	}
//	catch (const c10::Error& e) {
//		std::cerr << "error loading the model\n";
//		std::cerr << e.what();
//		return -1;
//	}
//	std::cout << "loading ok\n";
//
//	torch::DeviceType device_type = torch::kCUDA;
//	torch::Device device(device_type);
//	module.to(device);
//
//	//// 入力となるat::Tensorを作る [1 x 3 x 512 x 512]
//	//std::vector<torch::jit::IValue> input;
//	//input.push_back(torch::ones({ 1, 3, 256, 256 }).to(device));
//	//
//	//
//	////std::cout << "dim = [" << input.size(0)
//	////	<< " x " << input.size(1)
//	////	<< " x " << input.size(2)
//	////	<< " x " << input.size(3)
//	////	<< "]\n";
//
//	//// Moduleを実行し，出力を受け取る
//    at::Tensor output_tensor = torch::ones({ 1, 3, 256, 256 });
//	output_tensor.to(device);
//	//std::cout << "dim = [" << output_tensor.size(0)
//	//	<< " x " << output_tensor.size(1)
//	//	<< " x " << output_tensor.size(2)
//	//	<< " x " << output_tensor.size(3)
//	//	<< "]\n";
//	//try {
//	//	output_tensor = module.forward({ input }).toTensor();
//	//	//at::Tensor output_tensor = module.forward({ input }).toTensor();
//	//	// 出力Tensorの次元を確認 (DepthNetだと [1 x 128 x 128])
//	//	std::cout << "dim = [" << output_tensor.size(0)
//	//		<< " x " << output_tensor.size(1)
//	//		<< " x " << output_tensor.size(2)
//	//		<< " x " << output_tensor.size(3)
//	//		<< "]\n";
//
//
//
//	//}
//	//catch (const c10::Error& e) {
//	//	std::cerr << "error loading the model\n";
//	//	std::cerr << e.what();
//	//}
//	//std::cout << "pred ok\n";
//	//std::cout << "dim = [" << output_tensor.size(0)
//	//	<< " x " << output_tensor.size(1)
//	//	<< " x " << output_tensor.size(2)
//	//	<< " x " << output_tensor.size(3)
//	//	<< "]\n";
//
//	//at::Tensor imgtensor = torch::ones({ 1, 3, 256, 256 });
//	std::vector<torch::jit::IValue> imgtensor;
//
//	try {
//		//imgtensor = imgread(argv[2]);//00000016.png
//		imgtensor = imgread("./00000016.png");//
//		//std::cout << "dim imgtensor [" << imgtensor << "]\n";
//
//	}
//	catch (const c10::Error& e) {
//		std::cerr << "error imread\n";
//		std::cerr << e.what();
//	}
//	std::cout << "imread ok\n";
//
//	try {
//		std::vector<torch::jit::IValue> input;
//		input.push_back(torch::ones({ 1, 3, 256, 256 }).to(device));
//		output_tensor = module.forward({ input }).toTensor();
//		
//		//std::cout << "output_tensor [" << output_tensor << "]\n";
//		
//		output_tensor = module.forward({ imgtensor }).toTensor();
//		//std::cout << "output_tensor [" << output_tensor  << "]\n";
//
//		// 出力Tensorの次元を確認 (DepthNetだと [1 x 128 x 128])
//		std::cout << "dim = [" << output_tensor.size(0)
//			<< " x " << output_tensor.size(1)
//			<< " x " << output_tensor.size(2)
//			<< " x " << output_tensor.size(3)
//			<< "]\n";
//
//
//
//	}
//	catch (const c10::Error& e) {
//		std::cerr << "error forward the model\n";
//		std::cerr << e.what();
//	}
//	std::cout << "imread pred	ok\n";
//
//	try {
//
//		// 出力のat::Tensorをcv::Matへ格納
//		std::vector<int64_t> mydims{ static_cast<int64_t>(1), // 1
//						  static_cast<int64_t>(3), // 3
//						  static_cast<int64_t>(256), // h=512
//						  static_cast<int64_t>(256) }; // w=512
//		//output_tensor = output_tensor * 128.0 + 128.0;
//		output_tensor = output_tensor.permute({ 0, 2, 3, 1 });
//		output_tensor = output_tensor.squeeze(0).detach();
//		output_tensor = output_tensor.mul(128).to(torch::kU8);
//		output_tensor = output_tensor.add(128).to(torch::kU8);
//		output_tensor = output_tensor.to(torch::kCPU);
//		cv::Mat m_output(256, 256, CV_8UC3,  output_tensor.data<unsigned char>());
//		//std::cout << "output_tensor [" << output_tensor << "]\n";
//		//tensor2img(output_tensor);
//		//output_tensor = ((output_tensor * 128)+ 128);
//		
//		
//		std::cout << m_output.size() << std::endl;
//		std::cout << m_output.channels() << std::endl;
//		std::cout << m_output.type() << std::endl;
//	//sample_DisplayImage_Cpp(m_output);
//		getminmax(m_output);
//		//cv::Mat m_output_con;
//		//m_output.convertTo(m_output_con, CV_8UC3);
//
//
//
//
//		cv::imwrite("./pred.jpg", m_output);
//
//	}
//	catch (const c10::Error& e) {
//		std::cerr << "error save img\n";
//		std::cerr << e.what();
//	}
//	std::cout << "save img	ok\n";
//
//}

