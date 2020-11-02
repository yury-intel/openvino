// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/activation.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;

namespace CPULayerTestsDefinitions  {

typedef std::tuple<
        LayerTestsDefinitions::activationParams,
        CPUSpecificParams>
        ActivationLayerCPUTestParamSet;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj) {
        LayerTestsDefinitions::activationParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ActivationLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::activationParams>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 15, 0, 32768);
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::activationParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        std::string strExpectedPrc;
        if (Precision::BF16 == inPrc) {
            strExpectedPrc = "BF16";
        } else if (Precision::FP32 == inPrc) {
            strExpectedPrc = "FP32";
        }

        std::string isaType;
        if (with_cpu_x86_avx512f()) {
            isaType = "jit_avx512";
        } else if (with_cpu_x86_avx2()) {
            isaType = "jit_avx2";
        } else if (with_cpu_x86_sse42()) {
            isaType = "jit_sse42";
        } else {
            isaType = "ref";
        }
        selectedType = isaType + "_" + strExpectedPrc;

        InferenceEngine::Precision netPrecision;
        std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
        std::pair<ActivationTypes, std::vector<float>> activationDecl;
        std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = basicParamsSet;

        // Withing the test scope we don't need any implicit bf16 optimisations, so let's run the network as is.
        configuration.insert({PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO});

        activationType = activationDecl.first;
        auto constantsValue = activationDecl.second;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
        auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, shapes.second, constantsValue);
        activation->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params, "Activation");
    }
};

TEST_P(ActivationLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Eltwise");
}


namespace {
// list only types supported by eltwise
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sqrt,        {{}}},
        {Sigmoid,     {{}}},
        {Tanh,        {{}}},
        {Relu,        {{}}},
        {Gelu,        {{}}},
        {Exp,         {{}}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Elu,         {{0.1f}}},
        {Swish,       {{0.1f}}},
        {HSwish,      {{}}},
        {Mish,        {{}}}
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
        {PReLu, {{-0.01f}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50}, {{1}, {50}}},
        {{1, 128}, {{1}, {128}}},
};

std::vector<Precision> bf16InpOutPrc = {Precision::BF16, Precision::FP32};

const auto basicCases = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
            ::testing::Values(Precision::BF16),
            ::testing::ValuesIn(bf16InpOutPrc),
            ::testing::ValuesIn(bf16InpOutPrc),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(emptyCPUSpec)
);

const auto basicPreluCases = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
                ::testing::Values(Precision::BF16),
                ::testing::ValuesIn(bf16InpOutPrc),
                ::testing::ValuesIn(bf16InpOutPrc),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(emptyCPUSpec)
);

INSTANTIATE_TEST_CASE_P(Activation_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases, ActivationLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(Activation_Eltwise_Prelu_CPU_BF16, ActivationLayerCPUTest, basicPreluCases, ActivationLayerCPUTest::getTestCaseName);
} // namespace

} // namespace CPULayerTestsDefinitions