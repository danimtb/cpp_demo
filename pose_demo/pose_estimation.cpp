#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <vector>

const std::vector<std::pair<int, int>> skeleton = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, 
    {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

int main(int argc, const char* argv[]) {
    if (argc != 3) return -1;

    std::string model_path = argv[1];
    std::string video_source = argv[2];

    bool use_cuda = torch::cuda::is_available();
    torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU);
    
    if (!use_cuda) {
        std::cerr << "CUDA is required for this specific GpuMat implementation." << std::endl;
        return -1;
    }
    
    std::cout << "Using CUDA Device: " << device << std::endl;

    torch::jit::script::Module module;
    try {
        std::cout << "Loading model from: " << model_path << "..." << std::endl;
        module = torch::jit::load(model_path, device);
        module.eval();
        module.to(torch::kHalf); // FP16 Inference
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        // e.what() prints the full PyTorch JIT traceback and error description
        std::cerr << "\n[C10 ERROR] Failed to load the model.\n";
        std::cerr << "Libtorch Exception Details:\n" << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        // Catches standard C++ errors (like bad allocations)
        std::cerr << "\n[STD ERROR] An unexpected error occurred:\n" << e.what() << std::endl;
        return -1;
    }

    cv::VideoCapture cap(std::stoi(video_source), cv::CAP_ANY);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cv::Mat img;
    
    // Declare GpuMats outside the loop to reuse VRAM allocations
    cv::cuda::GpuMat gpu_img;
    cv::cuda::GpuMat gpu_resized;

    torch::NoGradGuard no_grad;

    while (cap.read(img)) {
        if (img.empty()) break;

        // 1. Upload raw frame
        gpu_img.upload(img);

        // 2. Preprocess
        cv::cuda::resize(gpu_img, gpu_resized, cv::Size(640, 640));
        cv::cuda::cvtColor(gpu_resized, gpu_resized, cv::COLOR_BGR2RGB);

        // 3. ZERO-COPY TRANSFER (FIXED: Handling CUDA memory padding/strides)
        auto options = torch::TensorOptions().dtype(torch::kByte).device(device);
        
        // OpenCV's 'step' is the true width of the row in bytes (including padding)
        // Strides are calculated in elements (which is 1 byte for kByte)
        std::vector<int64_t> strides = {
            (int64_t)gpu_resized.step * gpu_resized.rows, // Batch stride
            (int64_t)gpu_resized.step,                    // Row stride (Y)
            3,                                            // Pixel stride (X)
            1                                             // Channel stride (C)
        };
        
        torch::Tensor img_tensor = torch::from_blob(
            gpu_resized.data, 
            {1, 640, 640, 3}, 
            strides, 
            options
        );

        // 4. Format for PyTorch
        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous().to(torch::kHalf).div(255.0);

        // 5. Inference
        auto output_ivalue = module.forward({img_tensor});
        torch::Tensor preds = output_ivalue.isTensor() ? 
                              output_ivalue.toTensor() : 
                              output_ivalue.toTuple()->elements()[0].toTensor();

        // 6. Move to CPU (FIXED: Added .contiguous() to align memory for raw pointers)
        preds = preds.to(torch::kCPU).to(torch::kFloat32).squeeze(0).transpose(0, 1).contiguous();
        auto preds_a = preds.accessor<float, 2>();

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<std::vector<float>> keypoints_list;
        
        boxes.reserve(100); 
        scores.reserve(100);
        keypoints_list.reserve(100);

        for (int i = 0; i < preds.size(0); ++i) {
            float score = preds_a[i][4];
            if (score > 0.5f) {
                float cx = preds_a[i][0], cy = preds_a[i][1], w = preds_a[i][2], h = preds_a[i][3];
                boxes.emplace_back(cv::Rect(cx - w / 2, cy - h / 2, w, h));
                scores.push_back(score);
                
                // This raw pointer arithmetic now works safely because of .contiguous()
                std::vector<float> kpts(preds_a[i].data() + 5, preds_a[i].data() + 56);
                keypoints_list.push_back(std::move(kpts));
            }
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.4f, nms_indices);

        float x_scale = (float)img.cols / 640.0f;
        float y_scale = (float)img.rows / 640.0f;

        // 7. Draw on CPU cv::Mat
        for (int idx : nms_indices) {
            // Draw Bounding Box (added for easier debugging)
            cv::Rect scaled_box(
                boxes[idx].x * x_scale, 
                boxes[idx].y * y_scale, 
                boxes[idx].width * x_scale, 
                boxes[idx].height * y_scale
            );
            cv::rectangle(img, scaled_box, cv::Scalar(255, 0, 0), 2);

            // Draw Skeleton
            auto& kpts = keypoints_list[idx];
            for (const auto& bone : skeleton) {
                int a = bone.first, b = bone.second;
                if (kpts[a*3+2] > 0.5f && kpts[b*3+2] > 0.5f) {
                    cv::line(img, 
                             cv::Point(kpts[a*3]*x_scale, kpts[a*3+1]*y_scale),
                             cv::Point(kpts[b*3]*x_scale, kpts[b*3+1]*y_scale), 
                             cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        cv::imshow("CUDA Pose Demo", img);
        if (cv::waitKey(1) == 27) break;
    }
    
    return 0;
}
