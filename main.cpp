#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <map>

int main(int argc, char *argv[]) {
    // --- 1. Arguments ---
    std::string video_file = "../../assets/video1.mp4";
    std::string model_file = "../../models/MiDaS_small.pt";
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

    // --- 2. Load Model and Device ---
    torch::Device device(torch::kCPU); // Change to kCUDA when available
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

    cv::VideoCapture cap(video_file);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video/camera " << video_file << std::endl;
        return -1;
    }

    // Parameters from https://github.com/AbirKhan96/Intel-ISL-MiDaS
    const int TARGET_RES = 384;
    const int DISPLAY_WIDTH = 800;
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std_dev = {0.229f, 0.224f, 0.225f};

    cv::Mat frame, display_frame, input_resized;

    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();

        // A. Resize
        float aspect_ratio = (float)frame.cols / frame.rows;
        cv::resize(frame, display_frame, cv::Size(DISPLAY_WIDTH, DISPLAY_WIDTH / aspect_ratio));

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
        cv::resize(display_frame, input_resized, cv::Size(new_w, new_h));
        cv::cvtColor(input_resized, input_resized, cv::COLOR_BGR2RGB);
        input_resized.convertTo(input_resized, CV_32FC3, 1.0f / 255.0f);

        // Create CHW Tensor
        torch::Tensor tensor = torch::from_blob(input_resized.data, {1, new_h, new_w, 3}, torch::kFloat);
        tensor = tensor.permute({0, 3, 1, 2});

        // Normalize
        tensor[0][0].sub_(mean[0]).div_(std_dev[0]);
        tensor[0][1].sub_(mean[1]).div_(std_dev[1]);
        tensor[0][2].sub_(mean[2]).div_(std_dev[2]);
        
        tensor = tensor.to(device);

        // D. Inference
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = module.forward({tensor}).toTensor();
        }

        // E. Post-processing
        output = output.detach().squeeze().to(torch::kCPU);
        
        // Normalize visualization (OpenCV is quicker in CPU)
        cv::Mat depth_raw(new_h, new_w, CV_32FC1, output.data_ptr());
        cv::Mat depth_norm;
        cv::normalize(depth_raw, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        // F. Visualization
        cv::Mat depth_color;
        cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_MAGMA);
        cv::resize(depth_color, depth_color, display_frame.size());

        // G. Picture-in-Picture
        cv::Mat small_orig;
        int pip_w = depth_color.cols / 4;
        int pip_h = depth_color.rows / 4;
        cv::resize(display_frame, small_orig, cv::Size(pip_w, pip_h));

        int margin = 2;
        cv::Rect roi(margin, depth_color.rows - pip_h - margin, pip_w, pip_h);
        small_orig.copyTo(depth_color(roi));
        cv::rectangle(depth_color, roi, cv::Scalar(255, 255, 255), 1);

        // H. FPS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::string fps_text = "FPS: " + std::to_string((int)(1.0 / diff.count()));
        cv::putText(depth_color, fps_text, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);

        cv::imshow("LibTorch CPU Optimized - MiDaS v2.1", depth_color);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
