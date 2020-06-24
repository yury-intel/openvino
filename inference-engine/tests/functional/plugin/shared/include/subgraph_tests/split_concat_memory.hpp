// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

class SplitConcatMemory : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj);

protected:
    void SetUp() override;

    int axis;
    std::string net_xml;
    InferenceEngine::Blob::Ptr net_bin;
};

}  // namespace LayerTestsDefinitions