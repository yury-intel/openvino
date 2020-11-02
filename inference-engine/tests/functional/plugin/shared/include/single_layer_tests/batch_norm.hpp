// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
        double,                        // epsilon
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::Precision,    // Input precision
        InferenceEngine::Precision,    // Output precision
        InferenceEngine::Layout,       // Input layout
        InferenceEngine::Layout,       // Output layout
        InferenceEngine::SizeVector,   // Input shapes
        LayerTestsUtils::TargetDevice  // Target device name
> BatchNormLayerTestParams;

namespace LayerTestsDefinitions {

class BatchNormLayerTest : public testing::WithParamInterface<BatchNormLayerTestParams>,
                           public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions