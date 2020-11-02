// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ie_plugin_config.hpp>
#include <ie_core.hpp>
#include <functional>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "single_layer_tests/reshape.hpp"

namespace LayerTestsDefinitions {
    std::string ReshapeLayerTest::getTestCaseName(testing::TestParamInfo<reshapeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, outFormShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    bool specialZero;
    std::tie(specialZero, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outFormShapes, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "specialZero=" << specialZero << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ReshapeLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes, outFormShapes;
    bool specialZero;
    InferenceEngine::Precision netPrecision;
    std::tie(specialZero, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outFormShapes, targetDevice, configuration) =
        this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto constNode = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::Type_t::i64, ngraph::Shape{outFormShapes.size()}, outFormShapes);
    auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(
            std::make_shared<ngraph::opset1::Reshape>(paramIn[0], constNode, specialZero));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Reshape");
}

TEST_P(ReshapeLayerTest, CompareWithRefsDynamicBath) {
    Run();
}
}  // namespace LayerTestsDefinitions