// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "bfloat16_helpers.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {
namespace {
//    static const size_t inputSize = 40;
    static const size_t inputSize = 456;
}  // namespace

class Efficient_b5 : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//                   Add (FP32) - ScaleShift
//                    |
//                  Convolution (BF16)
//                    |
//                Add(fused)
//                 |       \
//                 |       Sigmoid (FP32)
//                 |       /
//               Mul(FP32) - Eltwise/Prod
//                    |
//                 DW Conv (BF16)
//                    |
//                   Add (fused)
        size_t outChannelsCount = 48;
        auto channelsCount = inputShapes[1];
        ngraph::Shape constShape = { 1, channelsCount, 1, 1 };
        ngraph::Shape outConstShape = { 1, outChannelsCount, 1, 1 };

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");
        // add
        std::shared_ptr<ngraph::opset1::Constant> const2 = nullptr;
        if (netPrecision == Precision::FP32) {
            std::vector<float> constValuesFP32;
            constValuesFP32.resize(1 * channelsCount * 1 * 1);
            FuncTestUtils::fillInputsBySinValues(constValuesFP32.data(), constValuesFP32.size());
            const2 = std::make_shared<ngraph::opset1::Constant>(ntype, constShape, constValuesFP32);
        } else {
            std::vector<short> constValuesBF16;
            constValuesBF16.resize(1 * channelsCount * 1 * 1);
            FuncTestUtils::fillInputsBySinValues(constValuesBF16.data(), constValuesBF16.size());
            const2 = std::make_shared<ngraph::opset1::Constant>(ntype, constShape, constValuesBF16.data());
        }
        auto addNode = std::make_shared<opset1::Add>(input1, const2);
        addNode->set_friendly_name("ADD_1");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { outChannelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(outChannelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(outChannelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
                addNode, weightsNode,
                ngraph::Strides({ 2, 2 }),   // strides
                ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("CONV_1");

        // add
        std::shared_ptr<ngraph::opset1::Constant> const3 = nullptr;
        if (netPrecision == Precision::FP32) {
            std::vector<float> constValuesFP32;
            constValuesFP32.resize(1 * outChannelsCount * 1 * 1);
            FuncTestUtils::fillInputsBySinValues(constValuesFP32.data(), constValuesFP32.size());
            const3 = std::make_shared<ngraph::opset1::Constant>(ntype, outConstShape, constValuesFP32);
        } else {
            std::vector<short> constValuesBF16;
            constValuesBF16.resize(1 * outChannelsCount * 1 * 1);
            FuncTestUtils::fillInputsBySinValues(constValuesBF16.data(), constValuesBF16.size());
            const3 = std::make_shared<ngraph::opset1::Constant>(ntype, outConstShape, constValuesBF16.data());
        }
        auto addNode2 = std::make_shared<opset1::Add>(convNode1, const3);
        addNode2->set_friendly_name("ADD_2");

        // sigmoid
        auto sigmNode =  std::make_shared<opset1::Sigmoid>(addNode2);
        sigmNode->set_friendly_name("SIGMOID");

        // multiply
        auto mulNode2 = std::make_shared<opset1::Multiply>(addNode2, sigmNode);
        mulNode2->set_friendly_name("MUL_2");

        // DW convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode2 = nullptr;
        ngraph::Shape convFilterShape2 = { outChannelsCount, 1, 1, 3, 3 };
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValues2FP32;
            weightValues2FP32.resize(outChannelsCount * 1 * 1 * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValues2FP32.data(), weightValues2FP32.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValues2FP32);
        } else {
            std::vector<short> weightValues2BF16;
            weightValues2BF16.resize(outChannelsCount * 1 * 1 * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValues2BF16.data(), weightValues2BF16.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValues2BF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::GroupConvolution>(
                mulNode2, weightsNode2,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("CONV_2");

        // add
        std::shared_ptr<ngraph::opset1::Constant> const4 = nullptr;
        if (netPrecision == Precision::FP32) {
            std::vector<float> constValuesFP32;
            constValuesFP32.resize(1 * outChannelsCount * 1 * 1);
            FuncTestUtils::fillInputsBySinValues(constValuesFP32.data(), constValuesFP32.size());
            const4 = std::make_shared<ngraph::opset1::Constant>(ntype, outConstShape, constValuesFP32);
        } else {
            std::vector<short> constValuesBF16;
            constValuesBF16.resize(1 * outChannelsCount * 1 * 1);
            FuncTestUtils::fillInputsBySinValues(constValuesBF16.data(), constValuesBF16.size());
            const4 = std::make_shared<ngraph::opset1::Constant>(ntype, outConstShape, constValuesBF16.data());
        }
        auto addNode3 = std::make_shared<opset1::Add>(convNode2, const4);
        addNode3->set_friendly_name("ADD_3");

        return std::make_shared<ngraph::Function>(addNode3, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE1:
        threshold = 0.4f;  // maximum value in tensor is 54.89
        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters

        expectedPrecisions["ADD_1"] = "FP32";
//        expectedPrecisions["ADD_2"] = "FP32";
//        expectedPrecisions["ADD_3"] = "FP32";
//        expectedPrecisions["CONV_1"] = "BF16";
//        expectedPrecisions["CONV_2"] = "BF16";
        expectedPrecisions["ADD_2"] = "BF16";
        expectedPrecisions["ADD_3"] = "BF16";
//        expectedPrecisions["SIGMOID"] = "ndef";
//        expectedPrecisions["MUL_1"] = "ndef";
//        expectedPrecisions["MUL_2"] = "ndef";
    }
};

TEST_P(Efficient_b5, CompareWithRefImpl) {
    test();
};

INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, Efficient_b5,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({ 1, 3, inputSize, inputSize })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Efficient_b5::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BF16_bfloat16_NoReshape, Efficient_b5,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({ 1, 3, inputSize, inputSize })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Efficient_b5::getTestCaseName);


}  // namespace LayerTestsDefinitions
