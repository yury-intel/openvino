// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/split_concat_memory.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
//        InferenceEngine::Precision::I64,
//        InferenceEngine::Precision::U64,
//        InferenceEngine::Precision::BF16,
//        InferenceEngine::Precision::U16,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
};

const std::vector<InferenceEngine::SizeVector> shapes = {
    {1, 8, 3, 2},
    {3, 8, 3, 2},
    {3, 8, 3},
    {3, 8},
};

INSTANTIATE_TEST_CASE_P(CPU, SplitConcatMemory,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        SplitConcatMemory::getTestCaseName);
}  // namespace




