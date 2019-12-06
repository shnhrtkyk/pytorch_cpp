#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
	if (argc != 2) {
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}


	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load(argv[1]);
		//module = torch::jit::load("model.pt");
	
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		std::cerr << e.what();
		return -1;
	}
	std::cout << "loading ok\n";

	torch::DeviceType device_type = torch::kCUDA;
	torch::Device device(device_type);
	module.to(device);

	// 入力となるat::Tensorを作る [1 x 6 x 512 x 512]
	std::vector<torch::jit::IValue> input;
	input.push_back(torch::ones({ 1, 3, 256, 256 }).to(device));
	//at::Tensor input = torch::ones({ 1, 3, 256, 256 });
	
	//std::cout << "dim = [" << input.size(0)
	//	<< " x " << input.size(1)
	//	<< " x " << input.size(2)
	//	<< " x " << input.size(3)
	//	<< "]\n";

	// Moduleを実行し，出力を受け取る
    at::Tensor output_tensor = torch::ones({ 1, 3, 256, 256 });
	output_tensor.to(device);
	std::cout << "dim = [" << output_tensor.size(0)
		<< " x " << output_tensor.size(1)
		<< " x " << output_tensor.size(2)
		<< " x " << output_tensor.size(3)
		<< "]\n";
	try {
		auto output_tensor = module.forward({ input }).toTensor();
		//at::Tensor output_tensor = module.forward({ input }).toTensor();
		// 出力Tensorの次元を確認 (DepthNetだと [1 x 128 x 128])



	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		std::cerr << e.what();
	}
	std::cout << "pred ok\n";
	//std::cout << "dim = [" << output_tensor.size(0)
	//	<< " x " << output_tensor.size(1)
	//	<< " x " << output_tensor.size(2)
	//	<< "]\n";

	
}
