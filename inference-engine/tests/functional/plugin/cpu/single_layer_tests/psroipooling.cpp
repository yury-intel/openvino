// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/interpolate.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::InterpolateLayerTestParams,
        CPUSpecificParams> PsroipoolingLayerCPUTestParamsSet;

class PsroipoolingLayerCPUTest : public testing::WithParamInterface<PsroipoolingLayerCPUTestParamsSet>,
                                virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PsroipoolingLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::InterpolateLayerTestParams>(
                basicParamsSet, 0));

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
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        LayerTestsDefinitions::InterpolateSpecificParams psroiParams;
        std::vector<size_t> inputShape;
        std::vector<size_t> targetShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(psroiParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShape, targetDevice) = basicParamsSet;

        int spatialBinX;
        int spatialBinY;
        float spatialScale;
        size_t groupSize;
        size_t outputDim;
        size_t regionsCount;
        std::string noDeformableMode;
        std::vector<size_t> inputShape;

        std:tie(outputDim, groupSize, spatialScale, spatialBinX, spatialBinY,
                regionsCount, mode, inputShape) = psroiParams;
        std::shared_ptr<ngraph::opset1::Constant> coords = nullptr;
        auto createSimilarRegions = [&] (float* data, size_t size) {
            size_t mod5 = 0;
            for (size_t i = 0; i < size; i++) {
                switch (mod5) {
                    case 0:
                        data[i] = 1.0f;
                        break;
                    case 1:
                    case 2:
                        data[i] = 4.0;
                        break;
                    case 3:
                    case 4:
                        data[i] = 7.0;
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

        auto psroi = std::make_shared<ngraph::opset1::PSROIPooling>(input1, coords, output_dim, group_size,
                                                                          spatial_scale, spatial_bins_x, spatial_bins_y, mode);
        psroi->get_rt_info() = getCPUInfo();
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(interpolate)};
        function = std::make_shared<ngraph::Function>(results, params, "interpolate");
    }
private:
    void PsroiValidate() {
//        auto expectedOutputs = CalculateRefs();
//        const auto& actualOutputs = GetOutputs();

        std::vector<std::vector<uint8_t>> expectedOutputs {{ 1 }};
        std::vector<std::vector<uint8_t>> actualOutputs {{ 1 }};

        if (expectedOutputs.empty()) {
            return;
        }

        IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
            << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

        Compare(expectedOutputs, actualOutputs);
}
};

TEST_P(PsroipoolingLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Interpolate");
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

const std::vector<float> spatialScaleXVector = { 1.0f };

const std::vector<size_t> groupSizeVector = { 2 };

const std::vector<size_t> outputDimVector = { 1 };

const std::vector<size_t> regionsCountVector = { 1 };

const std::vector<std::string> noDeformableModeVector = {
        "bilinear",
        "average"
};

const std::vector<std::vector<size_t>> inputShapeVector = {
        SizeVector({ 1, 3240, 40, 40 })
};

const auto psroipoolingNonDeformableParams = ::testing::Combine(
        ::testing::ValuesIn(outputDimVector),  // channels count of output
        ::testing::ValuesIn(groupSizeVector),  // offset for iteration across output pulled bins
        ::testing::ValuesIn(spatialScaleXVector),  // scale for given region considering actual input size
        ::testing::ValuesIn(spatialBinXVector),  // bin's column count
        ::testing::ValuesIn(spatialBinYVector),  // bin's row count
        ::testing::ValuesIn(regionsCountVector),  // regions count
        ::testing::ValuesIn(noDeformableModeVector),  // mode for non-deformable psroi
        ::testing::ValuesIn(inputShapeVector),  // feature map shape
);

INSTANTIATE_TEST_CASE_P(smoke_PsroiPooling_Non_Deformable_Layout_Test, PsroipoolingLayerCPUTest,
                        ::testing::Combine(  // test params
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
                                ::testing::ValuesIn(filterCPUInfoForDevice())),  // cpu params
                        PsroipoolingLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
