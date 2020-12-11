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
        std::string td;
        Precision netPr;
        ROIAlignSpecificParams roiPar;
        std::tie(roiPar, netPr, td) = basicParamsSet;
        std::tie(pooled_h, pooled_w, spatialScale, samplingRatio,
                 proposalVector, roiIdxVector, mode, inputShape) = roiPar;
        std::ostringstream result;
        result << "ROIAlignTest_";
        result << (netPr == Precision::FP32 ? "FP32" : "BF16") << "_";
        result << mode << "_";
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
        std::tie(roiAlignParams, netPrecision, targetDevice) = basicParamsSet;
        inPrc = outPrc = netPrecision;
        std::tie(pooled_h, pooled_w, spatialScale, samplingRatio,
                 proposalVector, roiIdxVector, mode, inputShape) = roiAlignParams;
        std::shared_ptr<ngraph::opset1::Constant> coords = nullptr;
        std::shared_ptr<ngraph::opset1::Constant> roisIdx = nullptr;

        ngraph::Shape coordsShape = { proposalVector.size() / 4, 4 };
        ngraph::Shape idxVectorShape = { roiIdxVector.size() };

        coords = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, coordsShape, proposalVector.data());
        roisIdx = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i32, idxVectorShape, roiIdxVector.data());

        auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});

        auto roialign = std::make_shared<ngraph::opset3::ROIAlign>(params[0], coords, roisIdx, pooled_h, pooled_w,
                                                                   samplingRatio, spatialScale, mode);
        roialign->get_rt_info() = getCPUInfo();
        selectedType = std::string("unknown_") + inPrc.name();
        threshold = 0.001f;
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
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, nc, x}, {nChw16c}, {}, {}});
    } else if (with_cpu_x86_avx2() || with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {}, {}});

        resCPUParams.push_back(CPUSpecificParams{{nChw8c, nc, x}, {nChw8c}, {}, {}});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, nc, x}, {nchw}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, nc, x}, {nhwc}, {}, {}});
    }
    return resCPUParams;
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16
};

const std::vector<int> spatialBinXVector = { 2 };

const std::vector<int> spatialBinYVector = { 2 };

const std::vector<float> spatialScaleVector = { 1.0f };

const std::vector<int> poolingRatioVector = { 7 };

const std::vector<std::string> modeVector = {
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

const auto roiAlignParams = ::testing::Combine(
        ::testing::ValuesIn(spatialBinXVector),       // bin's column count
        ::testing::ValuesIn(spatialBinYVector),       // bin's row count
        ::testing::ValuesIn(spatialScaleVector),      // scale for given region considering actual input size
        ::testing::ValuesIn(poolingRatioVector),      // pooling ratio for bin
        ::testing::ValuesIn(proposalsVector),         // coordinate vector: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        ::testing::ValuesIn(roisIdxVector),           // batch id's vector
        ::testing::ValuesIn(modeVector),              // pooling mode
        ::testing::ValuesIn(inputShapeVector)         // feature map shape
);

INSTANTIATE_TEST_CASE_P(smoke_ROIAlignLayoutTest, ROIAlignLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        roiAlignParams,
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice())),
                                ROIAlignLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
