#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configurar dispositivo a CPU
    torch::Device device(torch::kCPU);

    torch::jit::script::Module module;
    try {
        // Cargar el modelo
        module = torch::jit::load("../MiDaS_small.pt");
        module.to(device);
        module.eval();
        
        // Optimización para CPU: usa todos los núcleos disponibles
        at::set_num_threads(std::thread::hardware_concurrency());
        std::cout << "Modelo cargado en CPU usando " << std::thread::hardware_concurrency() << " hilos." << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "\n--- ERROR DE LIBTORCH ---" << std::endl;
        std::cerr << e.what() << std::endl; // ESTO IMPRIMIRÁ LA TRAZA COMPLETA
        return -1;
    }

    cv::VideoCapture cap("../video1.mp4");
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    while (cap.read(frame)) {
        // Pre-procesamiento
        cv::Mat input_resized;
        cv::resize(frame, input_resized, cv::Size(384, 384));
        cv::cvtColor(input_resized, input_resized, cv::COLOR_BGR2RGB);

        torch::Tensor tensor = torch::from_blob(input_resized.data, {1, 384, 384, 3}, torch::kByte);
        tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat).div(255.0);

        // Inferencia
        torch::Tensor output;
        try {
            output = module.forward({tensor}).toTensor();
        } catch (const std::exception& e) {
            std::cerr << "Error en inferencia: " << e.what() << std::endl;
            break;
        }

        // Post-procesamiento
        output = output.detach().squeeze();
        
        // Normalización manual para visualización
        float max_val = output.max().item<float>();
        float min_val = output.min().item<float>();
        output = (output - min_val) / (max_val - min_val) * 255.0;
        output = output.to(torch::kU8);

        cv::Mat depth_mat(384, 384, CV_8UC1, output.data_ptr());
        cv::applyColorMap(depth_mat, depth_mat, cv::COLORMAP_JET);
        cv::resize(depth_mat, depth_mat, frame.size());

        cv::imshow("Demo CPU (Debugging)", depth_mat);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}