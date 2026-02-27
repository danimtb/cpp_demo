#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    torch::Device device(torch::kCUDA);

    // 1. Cargar el nuevo modelo
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("depth_anything_v2.pt");
        module.to(device);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error cargando el modelo\n";
        return -1;
    }

    cv::VideoCapture cap(0);
    cv::Mat frame;

    // Media y Desviación estándar para normalización (ImageNet)
    const float mean[] = {0.485, 0.456, 0.406};
    const float std[] = {0.229, 0.224, 0.225};

    while (cap.read(frame)) {
        cv::Mat input_resized;
        cv::resize(frame, input_resized, cv::Size(518, 518));
        cv::cvtColor(input_resized, input_resized, cv::COLOR_BGR2RGB);

        // 2. Pre-procesamiento con Normalización
        torch::Tensor tensor = torch::from_blob(input_resized.data, {1, 518, 518, 3}, torch::kByte);
        tensor = tensor.permute({0, 3, 1, 2}).to(device).to(torch::kFloat).div(255.0);
        
        // Aplicar normalización: (x - mean) / std
        tensor[0][0].sub_(mean[0]).div_(std[0]);
        tensor[0][1].sub_(mean[1]).div_(std[1]);
        tensor[0][2].sub_(mean[2]).div_(std[2]);

        // 3. Inferencia
        torch::Tensor output = module.forward({tensor}).toTensor();

        // 4. Post-procesamiento
        output = output.detach().squeeze().to(torch::kCPU);
        
        // Normalización min-max para visualización
        double min_val, max_val;
        cv::Mat depth_mat(518, 518, CV_32FC1, output.data_ptr());
        cv::minMaxLoc(depth_mat, &min_val, &max_val);
        depth_mat.convertTo(depth_mat, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));

        // Aplicar color y redimensionar al tamaño original
        cv::applyColorMap(depth_mat, depth_mat, cv::COLORMAP_INFERNO); // El mapa "Inferno" es muy vistoso
        cv::resize(depth_mat, depth_mat, frame.size());

        // Mostrar original y profundidad juntos
        cv::Mat combined;
        cv::hconcat(frame, depth_mat, combined);
        cv::imshow("Depth Anything V2 + CUDA Demo", combined);

        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}