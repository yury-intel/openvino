// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

void cvMat2ieChwFp32Blob(const cv::Mat& image, InferenceEngine::Blob::Ptr& blob) {
    InferenceEngine::MemoryBlob::Ptr memoryBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    const auto blobData = memoryBlob->wmap().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    const size_t channels = blob->getTensorDesc().getDims()[0];
    const size_t height = blob->getTensorDesc().getDims()[1];
    const size_t width = blob->getTensorDesc().getDims()[2];
    const size_t numPixels = height * width;

    std::cout << "cvMat2ieChwFp32Blob():" << std::endl;
    std::cout << " tensorDescription: " << memoryBlob->getTensorDesc().getLayout() << " = [" << channels << ", " << height << ", " << width << "]" << std::endl;

    cv::Mat imageResized(image);
    cv::resize(image, imageResized, cv::Size(width, height));
    imageResized.convertTo(imageResized, CV_32F, 1. / 255.);
    cv::cvtColor(imageResized, imageResized, cv::COLOR_BGR2RGBA);

    float* rowPtr;
    for (size_t row = 0; row < height; ++row) {
        rowPtr = imageResized.ptr<float>(row);

        for (size_t col = 0; col < width; ++col) {
//            my_file << rowPtr[col * 4 + 2] << "\n";
//            my_file << rowPtr[col * 4 + 1] << "\n";
//            my_file << rowPtr[col * 4 + 0] << "\n";
            blobData[                row * width + col] = rowPtr[col * 4 + 2];
            blobData[    numPixels + row * width + col] = rowPtr[col * 4 + 1];
            blobData[2 * numPixels + row * width + col] = rowPtr[col * 4 + 0];
        }
    }
//    std::ofstream my_file("/localdisk/home/ygaydayc_l/models/custom_style/blob.txt");
//    size_t size = blob->size();
//    for (size_t i = 0; i < size; i++) {
//        my_file << blobData[i] << "\n";
//    }
//    my_file.close();
}

cv::Mat ieNchwFp32cvMat(const InferenceEngine::Blob::Ptr blob) {
    cv::Mat image;
    cv::Mat imageComponents[3];

    InferenceEngine::MemoryBlob::Ptr memoryBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    auto blobData = memoryBlob->rmap().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

    const InferenceEngine::TensorDesc& tensorOutput = memoryBlob->getTensorDesc();
    const size_t height = tensorOutput.getDims()[2];
    const size_t width = tensorOutput.getDims()[3];

    imageComponents[0] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_32F, reinterpret_cast<uint8_t*>(&blobData[0 * width * height]));
    imageComponents[1] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_32F, reinterpret_cast<uint8_t*>(&blobData[1 * width * height]));
    imageComponents[2] = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_32F, reinterpret_cast<uint8_t*>(&blobData[2 * width * height]));

    cv::merge(imageComponents, 3, image);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    image.convertTo(image, CV_8U, 255.);

    return image;
}

int main(int argc, char* argv[]) {
    const std::string device("CPU");
    const std::string path("/localdisk/home/ygaydayc_l/models/custom_style/");
    const std::string modelFilename("customStyleTransfer.onnx");
    const std::string styleFilename("style.jpg");
//    const std::vector<std::string> imagesFilenames({"dresden.jpg", "group.jpg", "ship.jpg"});
    const std::vector<std::string> imagesFilenames({ "group.jpg", "ship.jpg"});
    std::cout << "\nship with group + print scatters \n";

//    const std::vector<std::string> imagesFilenames({"ship.jpg"});
    const std::string inputPortImage("tensor.1");
    const std::string inputPortStyle("tensor");
    const std::string outputPort("935");

    // ========== 1. Create inference engine core ==========
    std::cout << "========== 1. Create inference engine core ==========" << std::endl;

    InferenceEngine::Core core;

    // ========== 2. Read model ==========
    std::cout << "========== 2. Read model ==========" << std::endl;

    InferenceEngine::CNNNetwork network = core.ReadNetwork(path + modelFilename);

    // ========== 3. Print inputs and outputs ==========
    std::cout << "========== 3. Print inputs and outputs ==========" << std::endl;

    std::cout << "Network input/output:" << std::endl;
    std::cout << " Name: " << network.getName() << std::endl;

    InferenceEngine::InputsDataMap inputsDataMap = network.getInputsInfo();
    InferenceEngine::OutputsDataMap outputsDataMap = network.getOutputsInfo();

    std::cout << " Inputs info(s): " << std::endl;
    for (std::pair<std::string, InferenceEngine::InputInfo::Ptr> inputData : inputsDataMap) {
        std::string inputName = inputData.first;
        InferenceEngine::InputInfo::Ptr inputInfo = inputData.second;

        std::cout << "  " << inputName << ", layout = " << inputInfo->getLayout() << ", precision = " << inputInfo->getPrecision() << std::endl;
    }

    std::cout << " Outputs info(s):" << std::endl;
    for (std::pair<std::string, InferenceEngine::DataPtr> outputData : outputsDataMap) {
        std::string outputName = outputData.first;
        InferenceEngine::DataPtr outputInfo = outputData.second;

        std::cout << "  " << outputName << ", layout = " << outputInfo->getLayout() << ", precision = " << outputInfo->getPrecision() << std::endl;
    }

    // ========== 4. Load model ==========
    std::cout << "========== 4. Load model ==========" << std::endl;

    InferenceEngine::ExecutableNetwork executableNetwork = core.LoadNetwork(network, device);

    // ========== 5. Create infer request ==========
    std::cout << "========== 5. Create infer request ==========" << std::endl;

    InferenceEngine::InferRequest::Ptr inferRequestPtr = executableNetwork.CreateInferRequestPtr();

    // ========== 6. Prepare inputs ==========
    std::cout << "========== 6. Prepare inputs ==========" << std::endl;

    InferenceEngine::Blob::Ptr styleBlob, imageBlob0, imageBlob1;
    InferenceEngine::Blob::Ptr imageBlob = inferRequestPtr->GetBlob(inputPortImage);

    cv::Mat style, image;

    styleBlob = inferRequestPtr->GetBlob(inputPortStyle);

    style = cv::imread(path + styleFilename);
    cvMat2ieChwFp32Blob(style, styleBlob);

    for (const std::string& imageFileName : imagesFilenames) {
        image = cv::imread(path + imageFileName);
        cvMat2ieChwFp32Blob(image, imageBlob);

//        // ========== 6.5 Copy input ===========
//
//        cv::Mat output_special = ieNchwFp32cvMat(imageBlob);
//
//        cv::imwrite(path + "_!!_" + imageFileName, output_special);

        // ========== 7. Do inference ==========
        std::cout << "========== 7. Do inference ==========" << std::endl;

        inferRequestPtr->Infer();

        // ========== 8. Handle output ==========
        std::cout << "========== 8. Handle output ==========" << std::endl;

        InferenceEngine::Blob::Ptr outputBlob;

        outputBlob = inferRequestPtr->GetBlob(outputPort);

        cv::Mat output = ieNchwFp32cvMat(outputBlob);

        cv::imwrite(path + "_" + imageFileName, output);
    }

    return 0;
}