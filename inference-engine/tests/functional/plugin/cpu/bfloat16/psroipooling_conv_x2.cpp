// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <utility>

#include <ie_core.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class Psroipooling_conv_x2 : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //        Conv (BF16)
        //          |
        //        PSROIPooling
        //          |
        //        Conv (BF16)

        ngraph::element::Type ntype = ngraph::element::f32;

        const size_t output_dim = 1;
        const size_t group_size = 2;
        const float spatial_scale = 1.0f;
        int spatial_bins_x = 2;
        int spatial_bins_y = 2;
        const std::string mode = "bilinear";

        size_t regionsCount = 1;

        auto input1 = std::make_shared<opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");

        // add
        auto const2 = opset1::Constant::create(ngraph::element::f32, Shape{1}, { 1.0f });
        auto addNode = std::make_shared<opset1::Add>(input1, const2);
        addNode->set_friendly_name("ADD_1");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;

        auto channelsCount = inputShapes[1];

        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        std::vector<float> weightValuesFP32;
        weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
        FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
        weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
                addNode, weightsNode,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("CONV_1");

        std::shared_ptr<ngraph::opset1::Constant> coords = nullptr;
        auto createSimilarRegions = [&] (float* data, size_t size) {
            size_t mod5 = 0;
            for (size_t i = 0; i < size; i++) {
                switch (mod5) {
                    case 0:
                        data[i] = 0.0f;
                        break;
                    case 1:
                    case 2:
                        data[i] = 4.0;
                        break;
                    case 3:
                    case 4:
                        data[i] = 8.0;
                        break;
                }
                mod5 = (mod5 + 1) % 5;
            }
        };
        ngraph::Shape coordsShape = { regionsCount, 5 };
        std::vector<float> coordsFP32;
        coordsFP32.resize(regionsCount * 5);
        createSimilarRegions(coordsFP32.data(), coordsFP32.size());
        coords = std::make_shared<ngraph::opset1::Constant>(ntype, coordsShape, coordsFP32.data());

        auto psroi_pool = make_shared<ngraph::opset1::PSROIPooling>(convNode1, coords, output_dim, group_size,
                                                                    spatial_scale, spatial_bins_x, spatial_bins_y, mode);
        psroi_pool->set_friendly_name("PSROI");

        // Convolution
        auto channelsCount2 = regionsCount;
        std::shared_ptr<ngraph::opset1::Constant> weightsNode2 = nullptr;

        ngraph::Shape convFilterShape2 = { channelsCount2, channelsCount2, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        std::vector<float> weightValuesFP32_2;
        weightValuesFP32_2.resize(channelsCount2 * channelsCount2 * 3 * 3);
        FuncTestUtils::fillInputsBySinValues(weightValuesFP32_2.data(), weightValuesFP32_2.size());
        weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValuesFP32_2);

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
                psroi_pool, weightsNode2,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
            ngraph::CoordinateDiff({ 1, 1 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("CONV_2");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{convNode2}, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);
        threshold = 1.0f;
    }
};

TEST_P(Psroipooling_conv_x2, CompareWithRefImpl) {
    test();
}

INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, Psroipooling_conv_x2,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({ 1, 10, 10, 10 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Psroipooling_conv_x2::getTestCaseName);
}  // namespace LayerTestsDefinitions
