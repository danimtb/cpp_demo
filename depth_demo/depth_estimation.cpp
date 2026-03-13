#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <chrono>
#include <map>


int main(int argc, char *argv[]) {
	std::cout << "CUDA is available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;

	int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();

	if (cuda_devices > 0) {
		std::cout << "Success: OpenCV has CUDA support! " 
			<< cuda_devices << " GPU(s) detected." << std::endl;

		// Optional: Print device info
		//cv::cuda::printCudaDeviceInfo(0); 
	}

	// --- 1. Arguments ---
	std::string video_file = "../assets/video1.mp4";
	std::string model_file = "../MiDaS_small.pt";
	int cameraIndex = 0;
	std::map<std::string, std::string> arguments;

	for (int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		if (arg.find("--") == 0) {
			size_t equal_sign_pos = arg.find("=");
			std::string key = arg.substr(0, equal_sign_pos);
			std::string value = equal_sign_pos != std::string::npos ? arg.substr(equal_sign_pos + 1) : "";
			arguments[key] = value;
		}
	}

	// Print arguments
	std::cout << "Arguments:" << std::endl;
	for (auto const& [key, val] : arguments) std::cout << key << " = " << val << std::endl;

	if (arguments.count("--video")) {
		video_file = arguments["--video"];
	}

	if (arguments.count("--model")) {
		model_file = arguments["--model"];
	}

	if (arguments.count("--camera")) {
		cameraIndex = std::stoi(arguments["--camera"]);
	}


	// --- 2. Load Model and Device ---
	torch::Device device(torch::kCUDA);
	torch::jit::script::Module module;

	try {
		module = torch::jit::load(model_file);
		module.to(device);
		module.eval();
		at::set_num_threads(std::thread::hardware_concurrency());
	} catch (const c10::Error& e) {
		std::cerr << "Error loading the model: " << e.what() << std::endl;
		return -1;
	}

	cv::VideoCapture cap;
	if (arguments.count("--video")) {
		cap = cv::VideoCapture(video_file);
		std::cout << "Using video file " << video_file << std::endl;
	} else {
		cap = cv::VideoCapture(cameraIndex, cv::CAP_ANY);
		//cap.open(deviceID, apiID);
		cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
	}
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open the video/camera" << std::endl;
		return -1;
	} else {
		std::cout << "Video capture initialized" << std::endl;
	}

	std::cout << "cv::currentUIFramework() returns " << cv::currentUIFramework() << std::endl;

	// Pre-allocate Mean and Std tensors on the GPU for vectorized normalization
	torch::Tensor mean_tensor = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, 3, 1, 1}).to(device);
	torch::Tensor std_tensor = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, 3, 1, 1}).to(device);

	// Parameters from https://github.com/AbirKhan96/Intel-ISL-MiDaS
	const int TARGET_RES = 384;
	const int DISPLAY_WIDTH = 800;
	//const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
	//const std::vector<float> std_dev = {0.229f, 0.224f, 0.225f};

	cv::Mat frame, display_frame, input_resized;
	cv::cuda::GpuMat gpu_frame, gpu_resized, gpu_rgb, gpu_float;
	cv::cuda::GpuMat gpu_pip, gpu_depth_resized, gpu_depth_norm;
	cv::Mat cpu_depth_resized, cpu_pip;

	while (cap.read(frame)) {
		auto start = std::chrono::high_resolution_clock::now();

		gpu_frame.upload(frame);
		// A. Resize
		float aspect_ratio = (float)frame.cols / frame.rows;
		//cv::resize(frame, display_frame, cv::Size(DISPLAY_WIDTH, DISPLAY_WIDTH / aspect_ratio));

		// B. Calculate dimensions in multiples of 32
		int new_h, new_w;
		if (frame.cols > frame.rows) {
			new_w = TARGET_RES;
			new_h = static_cast<int>(std::round((TARGET_RES / aspect_ratio) / 32.0)) * 32;
		} else {
			new_h = TARGET_RES;
			new_w = static_cast<int>(std::round((TARGET_RES * aspect_ratio) / 32.0)) * 32;
		}

		// C. Pre-process: Resize -> RGB -> Float -> Normalize
		cv::cuda::resize(gpu_frame, gpu_resized, cv::Size(new_w, new_h));
		cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);
		gpu_rgb.convertTo(gpu_float, CV_32FC3, 1.0f / 255.0f);


		if (!gpu_float.isContinuous()) {
			cv::cuda::GpuMat continuous_gpu_float;
			gpu_float.copyTo(continuous_gpu_float);
			gpu_float = continuous_gpu_float;
		}
		// Create CHW Tensor
		auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
		torch::Tensor tensor = torch::from_blob(
				gpu_float.ptr<float>(),     // Raw pointer to GPU memory
				{1, new_h, new_w, 3},       // Shape (NHWC)
				options
				);
		tensor = tensor.permute({0, 3, 1, 2}).sub_(mean_tensor).div_(std_tensor);


		// D. Inference
		torch::Tensor output;
		{
			torch::NoGradGuard no_grad;
			output = module.forward({tensor}).toTensor();
		}

		// E. Post-processing
		output = output.squeeze().contiguous();


		cv::cuda::GpuMat gpu_depth_raw(
				new_h,
				new_w,
				CV_32FC1,
				output.data_ptr<float>() // Raw pointer to GPU memory
				);

		// 3. Normalize on the GPU
		cv::cuda::normalize(gpu_depth_raw, gpu_depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

		// 4. Resize the depth map on the GPU to match the display size
		cv::cuda::resize(gpu_depth_norm, gpu_depth_resized, cv::Size(DISPLAY_WIDTH, DISPLAY_WIDTH / aspect_ratio));

		// 5. Download the processed, lightweight 8-bit image back to the CPU
		gpu_depth_resized.download(cpu_depth_resized);


		// F. Visualization

		// F. Visualization (CPU)
		cv::Mat depth_color;
		cv::applyColorMap(cpu_depth_resized, depth_color, cv::COLORMAP_MAGMA);

		//cv::resize(depth_color, depth_color, display_frame.size());

		// G. Picture-in-Picture (GPU Resizing)
		int pip_w = DISPLAY_WIDTH / 4;
		int pip_h = (DISPLAY_WIDTH / aspect_ratio) / 4;

		// Resize the original high-res frame to PiP size entirely on the GPU
		cv::cuda::resize(gpu_frame, gpu_pip, cv::Size(pip_w, pip_h));

		// Download only the tiny PiP image to the CPU (very fast)
		gpu_pip.download(cpu_pip);

		// Overlay the CPU PiP image onto the CPU Depth Map
		int margin = 2;
		cv::Rect roi(margin, depth_color.rows - pip_h - margin, pip_w, pip_h);
		cpu_pip.copyTo(depth_color(roi));
		cv::rectangle(depth_color, roi, cv::Scalar(255, 255, 255), 1);

		// H. FPS
		if (device.is_cuda()) {
			torch::cuda::synchronize();
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::string fps_text = "FPS: " + std::to_string((int)(1.0 / diff.count()));
		cv::putText(depth_color, fps_text, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);

		cv::imshow("LibTorch CPU Optimized - MiDaS v2.1", depth_color);
		if (cv::waitKey(1) == 27) break;
	}

	return 0;
}
