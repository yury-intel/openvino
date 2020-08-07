// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/convolution.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ConvolutionTransformationTestParams {
public:
    low_precision::LayerTransformation::Params transformationParams;
    ngraph::builder::subgraph::ConvolutionFunction::ActualValues actual;
    ngraph::builder::subgraph::ConvolutionFunction::ExpectedValues expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    bool,
    ConvolutionTransformationTestParams> ConvolutionTransformationParams;

class ConvolutionTransformation : public LayerTransformation, public testing::WithParamInterface<ConvolutionTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const bool updatePrecisions = std::get<2>(GetParam());
        const ConvolutionTransformationTestParams testParams = std::get<3>(GetParam());

        const low_precision::LayerTransformation::Params params = low_precision::LayerTransformation::Params(testParams.transformationParams).
            setUpdatePrecisions(updatePrecisions);

        actualFunction = ngraph::builder::subgraph::ConvolutionFunction::getOriginal(
            precision,
            shape,
            params.updatePrecisions,
            testParams.actual);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConvolutionTransformation, ngraph::opset1::Convolution>(params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConvolutionFunction::getReference(
            precision,
            shape,
            params.updatePrecisions,
            testParams.expected);

        auto pr = actualFunction->get_parameters()[0]->get_element_type();
        size_t byteVectorSize = shape_size(shape);
        if (pr == ngraph::element::f32) {
            byteVectorSize *= 4;
        } else if (pr == ngraph::element::f16) {
            byteVectorSize *= 2;
        }

        for (auto inputCount = actualFunction->get_parameters().size(); inputCount != 0; inputCount--) {
            std::vector<std::uint8_t> inputVector(byteVectorSize);
            referenceInputs.push_back(inputVector);
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
        ngraph::element::Type precision;
        ngraph::Shape shape;
        bool updatePrecisions;
        ConvolutionTransformationTestParams params;
        std::tie(precision, shape, updatePrecisions, params) = obj.param;

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, params.transformationParams.setUpdatePrecisions(updatePrecisions)) <<
            params.actual << params.expected;
        return result.str();
    }
};

TEST_P(ConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST_P(ConvolutionTransformation, CompareOutputs) {
//    actualFunction->validate_nodes_and_infer_types();

    auto res = compareResults(actualFunction, referenceFunction);
    ASSERT_TRUE(res) << "Hello world\n";
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<bool> updatePrecisions = {
    true,
    false
};

const std::vector<ConvolutionTransformationTestParams> testParams = {
    // with zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { 128 },
            { 0.02f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { 128 },
            ngraph::element::i8,
            { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
            { },
            { 0.0002f }  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },
    // without zero point
    {
        LayerTransformation::createParamsU8I8(),
        // ActualValues
        {
            ngraph::element::u8,
            { },
            { 0.02f },
            { 2.f },
            { 255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f} }
        },
        // ExpectedValues
        {
            ngraph::element::u8,
            { },
            ngraph::element::i8,
            { -125.f }, // 2 (in: 0 - 254) => -125 (out: -127 - 127)
            { },
            { 0.0002f }  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    ConvolutionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(updatePrecisions),
        ::testing::ValuesIn(testParams)),
    ConvolutionTransformation::getTestCaseName);
