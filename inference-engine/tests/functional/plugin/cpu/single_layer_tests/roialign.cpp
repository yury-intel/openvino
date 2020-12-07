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
namespace {
    int pooled_h;
    int pooled_w;
    float spatialScale;
    int samplingRatio;
    std::vector<float> proposalVector;
    std::vector<size_t> roiIdxVector;
    std::string mode;
    std::vector<size_t> inputShape;
}  // namespace

typedef std::tuple<
        int,                  // bin's column count
        int,                  // bin's row count
        float,                // scale for given region considering actual input size
        int,                  // pooling ratio
        std::vector<float>,   // coordinate vector: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        std::vector<size_t>,  // batch id's vector
        std::string,          // pooling mode
        std::vector<size_t>   // feature map shape
> ROIAlignSpecificParams;

typedef std::tuple<
        ROIAlignSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shapes
        InferenceEngine::SizeVector,    // Target shapes
        LayerTestsUtils::TargetDevice   // Device name
> ROIALignLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::ROIALignLayerTestParams,
        CPUSpecificParams> ROIAlignLayerCPUTestParamsSet;

class ROIAlignLayerCPUTest : public testing::WithParamInterface<ROIAlignLayerCPUTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIAlignLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::ROIALignLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::ostringstream result;
        result << "ROIAlignTest_";
        result << std::to_string(obj.index);
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }
protected:
    void SetUp() override {
        CPULayerTestsDefinitions::ROIALignLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::ROIAlignSpecificParams roiAlignParams;
        std::vector<size_t> inputShape_unused;
        std::vector<size_t> targetShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(roiAlignParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape_unused, targetShape, targetDevice) = basicParamsSet;
        std::tie(pooled_h, pooled_w, spatialScale, samplingRatio,
                 proposalVector, roiIdxVector, mode, inputShape) = roiAlignParams;
        std::shared_ptr<ngraph::opset1::Constant> coords = nullptr;
        std::shared_ptr<ngraph::opset1::Constant> roisIdx = nullptr;

        ngraph::Shape coordsShape = { proposalVector.size() / 4, 4 };
        ngraph::Shape idxVectorShape = { roiIdxVector.size() };

        ngraph::element::Type ntype = inPrc == InferenceEngine::Precision::FP32 ? ngraph::element::f32 : ngraph::element::bf16;
        coords = std::make_shared<ngraph::opset1::Constant>(ntype, coordsShape, proposalVector.data());
        roisIdx = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, idxVectorShape, roiIdxVector.data());

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto roialign = std::make_shared<ngraph::opset3::ROIAlign>(params[0], coords, roisIdx, pooled_h, pooled_w,
                                                                   samplingRatio, spatialScale, mode);
        roialign->get_rt_info() = getCPUInfo();
        if (Precision::BF16 == netPrecision) {
            selectedType = "unknown_BF16";
        } else if (Precision::FP32 == netPrecision) {
            selectedType = "unknown_FP32";
        }

        threshold = 0.0f;
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(roialign)};
        function = std::make_shared<ngraph::Function>(results, params, "ROIAlign");
    }
};

TEST_P(ROIAlignLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "ROIAlign");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
//        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {}, {}});
//        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
//    } else if (with_cpu_x86_avx2()) {
//        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"});
//        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc}, {"jit_avx2"}, "jit_avx2_FP32"});
//        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
//    } else if (with_cpu_x86_sse42()) {
//        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"});
//        resCPUParams.push_back(CPUSpecificParams{{nhwc, x}, {nhwc}, {"jit_sse42"}, "jit_sse42_FP32"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x}, {nchw}, {"ref"}, "ref_FP32"});
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<int> spatialBinXVector = { 2 };

const std::vector<int> spatialBinYVector = { 2 };

const std::vector<float> spatialScaleVector = { 1.0f };

const std::vector<int> poolingRatioVector = { 2 };

const std::vector<std::string> noDeformableModeVector = {
        "avg",
        "max"
};

const std::vector<std::vector<size_t>> inputShapeVector = {
        SizeVector({ 1, 10, 20, 20 })
};

const std::vector<std::vector<float>> proposalsVector = {
        { 1, 1, 19, 19 }
};

const std::vector<std::vector<size_t>> roisIdxVector = {
        { 0 }
};

const auto roiAlignNonDeformableParams = ::testing::Combine(
        ::testing::ValuesIn(spatialBinXVector),       // bin's column count
        ::testing::ValuesIn(spatialBinYVector),       // bin's row count
        ::testing::ValuesIn(spatialScaleVector),      // scale for given region considering actual input size
        ::testing::ValuesIn(poolingRatioVector),      // pooling ratio for bin
        ::testing::ValuesIn(proposalsVector),         // coordinate vector: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        ::testing::ValuesIn(roisIdxVector),           // batch id's vector
        ::testing::ValuesIn(noDeformableModeVector),  // pooling mode
        ::testing::ValuesIn(inputShapeVector)         // feature map shape
);

const auto bigCombine = ::testing::Combine(
        ::testing::Combine(
                roiAlignNonDeformableParams,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 1, 40, 40})),
                ::testing::Values(std::vector<size_t>({1, 1, 50, 60})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice()));

INSTANTIATE_TEST_CASE_P(smoke_ROIAlignLayoutTest, ROIAlignLayerCPUTest,
                        bigCombine, ROIAlignLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
