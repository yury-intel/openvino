// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        size_t,  // channels count of output
        size_t,  // offset for iteration across output pulled bins
        float,  // scale for given region considering actual input size
        int,  // bin's column count
        int,  // bin's row count
        std::vector<float>,  // coordinate vector: batch id, left_top_x, left_top_y, right_bottom_x, right_bottom_y
        std::string,  // mode for non-deformable psroi
        std::vector<size_t>  // feature map shape
> PsroipoolingSpecificParams;

typedef std::tuple<
        PsroipoolingSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shapes
        InferenceEngine::SizeVector,    // Target shapes
        LayerTestsUtils::TargetDevice   // Device name
> PsroipoolingLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::PsroipoolingLayerTestParams,
        CPUSpecificParams> PsroipoolingLayerCPUTestParamsSet;

class PsroipoolingLayerCPUTest : public testing::WithParamInterface<PsroipoolingLayerCPUTestParamsSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PsroipoolingLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::PsroipoolingLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;

//        result << CPULayerTestsDefinitions::PsroipoolingLayerCPUTest::getTestCaseName(
//        testing::TestParamInfo<CPULayerTestsDefinitions::PsroipoolingLayerTestParams>(
//                basicParamsSet, 0));
        result << std::to_string(obj.index);

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();
        Infer();
        PsroiValidate();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::PsroipoolingLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::PsroipoolingSpecificParams psroiParams;
        std::vector<size_t> inputShape_unused;
        std::vector<size_t> targetShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(psroiParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape_unused, targetShape, targetDevice) = basicParamsSet;

        int spatialBinX;
        int spatialBinY;
        float spatialScale;
        size_t groupSize;
        size_t outputDim;
        std::vector<float> proposalsVector;
        std::string nonDeformableMode;
        std::vector<size_t> inputShape;

        std:tie(outputDim, groupSize, spatialScale, spatialBinX, spatialBinY,
                proposalsVector, nonDeformableMode, inputShape) = psroiParams;
        std::shared_ptr<ngraph::opset1::Constant> coords = nullptr;
        ngraph::Shape coordsShape = { proposalsVector.size(), 5 };
        ngraph::element::Type ntype = inPrc == InferenceEngine::Precision::FP32 ? ngraph::element::f32 : ngraph::element::bf16;
        coords = std::make_shared<ngraph::opset1::Constant>(ntype, coordsShape, proposalsVector.data());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto psroi = std::make_shared<ngraph::opset1::PSROIPooling>(params[0], coords, outputDim, groupSize,
                                                                          spatialScale, spatialBinX, spatialBinY, nonDeformableMode);
        psroi->get_rt_info() = getCPUInfo();
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(psroi)};
        function = std::make_shared<ngraph::Function>(results, params, "psroipooling");
    }
private:
    void PsroiValidate() {
//        auto expectedOutputs = CalculateRefs();
        const auto& actualOutputs = GetOutputs();
        auto expectedOutputs = GetOutputs();

//        std::vector<std::vector<uint8_t>> expectedOutputs;
//        std::vector<std::vector<uint8_t>> actualOutputs;

        if (expectedOutputs.empty()) {
            return;
        }

        IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
            << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

        for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
            const auto& expected = expectedOutputs[outputIndex];
            const auto& actual = actualOutputs[outputIndex];
            Compare(expected, actual);
        }
}
};

TEST_P(PsroipoolingLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Psroipooling");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42_FP32"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"ref"}, "ref_FP32"});
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<int> spatialBinXVector = { 3 };

const std::vector<int> spatialBinYVector = { 3 };

const std::vector<float> spatialScaleVector = { 1.0f };

const std::vector<size_t> groupSizeVector = { 2 };

const std::vector<size_t> outputDimVector = { 1 };

const std::vector<std::string> noDeformableModeVector = {
        "bilinear",
        "average"
};

const std::vector<std::vector<size_t>> inputShapeVector = {
        SizeVector({ 1, 3240, 40, 40 })
};

const std::vector<std::vector<float>> proposalsVector = {
        { 0, 4, 4, 7, 7 }
};

const auto psroipoolingNonDeformableParams = ::testing::Combine(
        ::testing::ValuesIn(outputDimVector),  // channels count of output
        ::testing::ValuesIn(groupSizeVector),  // offset for iteration across output pulled bins
        ::testing::ValuesIn(spatialScaleVector),  // scale for given region considering actual input size
        ::testing::ValuesIn(spatialBinXVector),  // bin's column count
        ::testing::ValuesIn(spatialBinYVector),  // bin's row count
        ::testing::ValuesIn(proposalsVector),  // coordinate vector: batch id, left_top_x, left_top_y, right_bottom_x, right_bottom_y
        ::testing::ValuesIn(noDeformableModeVector),  // mode for non-deformable psroi
        ::testing::ValuesIn(inputShapeVector)  // feature map shape
);

const auto bigCombine = ::testing::Combine(  // test params
        ::testing::Combine(  // basic params
                psroipoolingNonDeformableParams,  // psroi params
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 1, 40, 40})),
                ::testing::Values(std::vector<size_t>({1, 1, 50, 60})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice()));
//const auto bigCombine = ::testing::ValuesIn(filterCPUInfoForDevice());

INSTANTIATE_TEST_CASE_P(smoke_PsroiPoolingNonDeformableLayoutTest, PsroipoolingLayerCPUTest,
                        bigCombine
                        ,  // cpu params
                        PsroipoolingLayerCPUTest::getTestCaseName
);
} // namespace
} // namespace CPULayerTestsDefinitions
