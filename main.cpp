#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>


int main(int argc, char *argv[]) {

    std::string video_file = "../assets/video1.mp4";

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

    std::cout << arguments << std::endl;

    if (arguments.count("--video")) {
        video_file = arguments["--video"];
    }

    torch::Device device(torch::kCPU);
    torch::jit::script::Module module;

    try {
        module = torch::jit::load("../MiDaS_small.pt");
        module.to(device);
        module.eval();
        at::set_num_threads(std::thread::hardware_concurrency());
    } catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    cv::VideoCapture cap(video_file);
    if (!cap.isOpened()) return -1;

    // OPT 1: Resoluciones más ligeras
    const int INFERENCE_RES = 256; // Bajamos de 384 a 256 (55% menos píxeles)
    const int DISPLAY_WIDTH = 800; // Resolución fija de ventana para no saturar CPU

    cv::Mat frame, input_resized, display_frame;

    // while(cap.read(frame)) {  // Show unprocessed video
    //     auto start = std::chrono::high_resolution_clock::now();

    //     float aspect_ratio = (float)frame.cols / frame.rows;
    //     cv::resize(frame, display_frame, cv::Size(DISPLAY_WIDTH, DISPLAY_WIDTH / aspect_ratio));

    //     // Cálculo de FPS
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> diff = end - start;
    //     std::string fps = "FPS: " + std::to_string((int)(1.0 / diff.count()));
    //     cv::putText(display_frame, fps, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);

    //     cv::imshow("LibTorch CPU Optimized", display_frame);
    //     if (cv::waitKey(1) == 27) break;
    // }

    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();

        // OPT 2: Redimensionar el frame original de entrada inmediatamente
        // Esto hace que todo el dibujo posterior sea mucho más rápido
        float aspect_ratio = (float)frame.cols / frame.rows;
        cv::resize(frame, display_frame, cv::Size(DISPLAY_WIDTH, DISPLAY_WIDTH / aspect_ratio));

        // Pre-procesamiento
        cv::resize(display_frame, input_resized, cv::Size(INFERENCE_RES, INFERENCE_RES));
        cv::cvtColor(input_resized, input_resized, cv::COLOR_BGR2RGB);

        torch::Tensor tensor = torch::from_blob(input_resized.data, {1, INFERENCE_RES, INFERENCE_RES, 3}, torch::kByte);
        tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat).div(255.0);

        // OPT 3: Inferencia con NoGradGuard (Vital en CPU)
        torch::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = module.forward({tensor}).toTensor();
        }

        // OPT 4: Post-procesamiento rápido
        output = output.detach().squeeze();
        
        // Usamos normalize de OpenCV que suele estar mejor vectorizado para CPU que tensor.max()
        cv::Mat depth_raw(INFERENCE_RES, INFERENCE_RES, CV_32FC1, output.data_ptr());
        cv::Mat depth_norm;
        cv::normalize(depth_raw, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        // Visualización
        cv::Mat depth_color;
        cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_MAGMA);
        cv::resize(depth_color, depth_color, display_frame.size());

        // 7. Picture-in-Picture (Original en la esquina inferior izquierda)
        cv::Mat small_orig;
        int pip_w = display_frame.cols / 4;
        int pip_h = display_frame.rows / 4;
        cv::resize(display_frame, small_orig, cv::Size(pip_w, pip_h));

        // Definir ROI y copiar
        int margin = 2;
        cv::Rect roi(margin, depth_color.rows - pip_h - margin, pip_w, pip_h);
        small_orig.copyTo(depth_color(roi));
        cv::rectangle(depth_color, roi, cv::Scalar(255, 255, 255), 1); // Borde estético

        // Cálculo de FPS
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::string fps = "FPS: " + std::to_string((int)(1.0 / diff.count()));
        cv::putText(depth_color, fps, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);

        cv::imshow("LibTorch CPU Optimized", depth_color);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
