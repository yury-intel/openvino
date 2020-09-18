// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {
namespace {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
}  // namespace
class LSTM : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//              LSTM (FP32)

        auto channelsCount = inputShapes[1];

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");

//        auto X = make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size});
        std::shared_ptr<ngraph::opset1::Constant> R = nullptr;  // recurrence weights
        auto H_t = make_shared<ngraph::opset1::Parameter>(ntype,
                Shape{batch_size, hidden_size});  // initial_hidden_state
        auto C_t = make_shared<ngraph::opset1::Parameter>(ntype,
                Shape{batch_size, hidden_size});  // initial_cell_state
        std::shared_ptr<ngraph::opset1::Constant> W = nullptr;  // gate weights
//        const auto lstm_cell = make_shared<op::v0::LSTMCell>(X, H_t, C_t, W, R, hidden_size);
        ngraph::Shape recShape = { gates_count * hidden_size, hidden_size };
        ngraph::Shape gateShape = { gates_count * hidden_size, input_size };

        if (netPrecision == Precision::FP32) {
            std::vector<float> recWeightValuesFP32;
            std::vector<float> gateWeightValuesFP32;

            recWeightValuesFP32.resize(gates_count * hidden_size * hidden_size);
            gateWeightValuesFP32.resize(gates_count * hidden_size * input_size);

            FuncTestUtils::fillInputsBySinValues(recWeightValuesFP32.data(), recWeightValuesFP32.size());
            FuncTestUtils::fillInputsBySinValues(gateWeightValuesFP32.data(), gateWeightValuesFP32.size());

            R = std::make_shared<ngraph::opset1::Constant>(ntype, recShape, recWeightValuesFP32);
            W = std::make_shared<ngraph::opset1::Constant>(ntype, gateShape, gateWeightValuesFP32);
        } else {
            std::vector<short> recWeightValuesBF16;
            std::vector<short> gateWeightValuesBF16;

            recWeightValuesBF16.resize(gates_count * hidden_size * hidden_size);
            gateWeightValuesBF16.resize(gates_count * hidden_size * input_size);

            FuncTestUtils::fillInputsBySinValues(recWeightValuesBF16.data(), recWeightValuesBF16.size());
            FuncTestUtils::fillInputsBySinValues(gateWeightValuesBF16.data(), gateWeightValuesBF16.size());

            R = std::make_shared<ngraph::opset1::Constant>(ntype, recShape, recWeightValuesBF16);
            W = std::make_shared<ngraph::opset1::Constant>(ntype, gateShape, gateWeightValuesBF16);
        }

        std::shared_ptr<ngraph::Node> lstmNode = std::make_shared<ngraph::opset1::LSTMCell>(input1, H_t, C_t, W, R, hidden_size);
        lstmNode->set_friendly_name("LSTM");
        return std::make_shared<ngraph::Function>(lstmNode, ngraph::ParameterVector{input1, H_t, C_t});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2: set up safe threshold <= 5% from maximum value of output tensor

        // 256 channels
        // threshold = 0.26f;  // Max in fp32 network by output: 5.26852

        // 3 channels
        threshold = 0.2f;  // Max in fp32 network by output: 4.90418

        // STAGE3:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
//        expectedPrecisions["Convolution_0"] = "BF16";
//        expectedPrecisions["Convolution_1"] = "BF16";
//        expectedPrecisions["Elt_sum"] = "FP32";
    }
};

TEST_P(LSTM, CompareWithRefImpl) {
    test();
};

INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, LSTM,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({batch_size, input_size})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LSTM::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BF16_bfloat16_NoReshape, LSTM,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({batch_size, input_size})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        LSTM::getTestCaseName);

}  // namespace LayerTestsDefinitions
