// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <details/ie_exception.hpp>
#include "simple_low_precision_transformer.hpp"
#include <ngraph_functions/pass/convert_prc.hpp>

using namespace testing;
using namespace ngraph::pass;

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8U8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8 },
        { ngraph::element::u8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsI8I8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::i8 },
        { ngraph::element::i8 });
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8AndI8() {
    return low_precision::LayerTransformation::Params(
        true,
        low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        low_precision::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        { ngraph::element::u8, ngraph::element::i8 },
        { ngraph::element::i8 });
}

std::string LayerTransformation::toString(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric_" : "symmetric_") <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.precisionsOnActivations[0] << "_" <<
        params.precisionsOnWeights[0] << "_" <<
        params.quantizedTensorAlignmentOnActivations;

    return result.str();
}

void LayerTransformation::transform(std::shared_ptr<ngraph::Function> function) {
    ngraph::pass::low_precision::LowPrecisionTransformations transformations = ngraph::pass::low_precision::LowPrecisionTransformer::getAllTransformations();
    ngraph::pass::low_precision::LowPrecisionTransformer transformer(transformations);
    transformer.transform(function);
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ngraph::element::Type& type,
    const ngraph::Shape& shape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << type << "_" << shape << "_" << toString(params);
    return result.str();
}

template<class T>
void LayerTransformation::compareTypedVectors(
        const std::vector<std::vector<T>>& v1,
        const std::vector<std::vector<T>>& v2) {
    ASSERT_EQ(v1.size(), v2.size());
    auto size = v1.size();
    for (std::size_t idx = 0; idx < size; ++idx) {
        const auto& expected = v1[idx];
        const auto& actual = v2[idx];
        if ( !std::is_same<T, int>::value &&
        !std::is_same<T, float>::value) {
            THROW_IE_EXCEPTION << "Precision not supported";
        }
        compareValues<T>(v1[idx].data(), v2[idx].data(), v1.size());
    }
}

template<class T>
void LayerTransformation::compareBytes(
        const std::vector<std::vector<std::uint8_t>>& expectedVector,
        const std::vector<std::vector<std::uint8_t>>& actualVector) {
    for (std::size_t idx = 0; idx < expectedVector.size(); ++idx) {
        const auto& expected = expectedVector[idx];
        const auto& actual = actualVector[idx];
        ASSERT_EQ(expectedVector.size(), actualVector.size());
        const unsigned char *expectedBuffer = expected.data();
        const unsigned char *actualBuffer = actual.data();
        auto size = actual.size();
        compareValues<T>(expectedBuffer, actualBuffer, size);
    }
}

template<class T>
void LayerTransformation::compareValues(const T *expected, const T *actual, std::size_t size, T thr) {
    std::cout << std::endl;
    for (std::size_t i = 0; i < size; ++i) {
        const auto &ref = expected[i];
        const auto &res = actual[i];
        const auto absoluteDifference = std::abs(res - ref);
        if (absoluteDifference <= thr) {
            continue;
        }

        const auto max = std::max(std::abs(res), std::abs(ref));
        ASSERT_TRUE(max != 0 && ((absoluteDifference / max) <= thr))
                                    << "Relative comparison of values expected: " << ref << " and actual: " << res
                                    << " at index " << i << " with t " << thr
                                    << " failed";
    }
}

template<class T>
void LayerTransformation::compareValues(const void *expected, const void *actual, std::size_t size) {
    if (std::is_same<T, float>::value) {
        compareValues(
                reinterpret_cast<const float *>(expected), reinterpret_cast<const float *>(actual),
                size, threshold);
    } else if (std::is_same<T, int>::value) {
        compareValues(
                reinterpret_cast<const std::int32_t *>(expected),
                reinterpret_cast<const std::int32_t *>(actual), size, 0);
    } else {
        FAIL() << "Comparator for given precision isn't supported";
    }
}
std::vector<std::vector<std::uint8_t>> LayerTransformation::CalculateRefs(
        std::shared_ptr<ngraph::Function> _function,
        std::vector<std::vector<std::uint8_t>> _inputs) {
    // nGraph interpreter does not support f16
    // IE converts f16 to f32
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(_function);
    _function->validate_nodes_and_infer_types();
    return ngraph::helpers::interpreterFunction(_function, _inputs, ::ngraph::element::Type_t::undefined);
}

bool LayerTransformation::compareResults(std::shared_ptr<ngraph::Function> f1, std::shared_ptr<ngraph::Function> f2) {
    auto pr = actualFunction->get_parameters()[0]->get_element_type();
    auto sh = actualFunction->get_parameters()[0]->get_partial_shape().to_shape();
    size_t byteVectorSize = shape_size(sh);
    if (pr == ngraph::element::f32) {
        byteVectorSize *= 4;
    } else if (pr == ngraph::element::f16) {
        byteVectorSize *= 2;
    }
    const auto &parameters = actualFunction->get_parameters();
    for (const auto &parameter : parameters) {
        const auto &parameterIndex = actualFunction->get_parameter_index(parameter);
        const auto &parameterShape = parameter->get_shape();
        const auto &parameterType = parameter->get_element_type();
        const auto &parameterSize = shape_size(parameterShape) * parameterType.size();
        std::vector<std::uint8_t> inputVector(parameterSize);
        byteInputData.push_back(inputVector);
    }

    auto type = actualFunction->get_parameters()[0]->get_element_type();
    if (type == ::ngraph::element::f32) {
        FuncTestUtils::fillByteVectorAsTyped<float>(byteInputData);
    } else if (type == ::ngraph::element::i32) {
        FuncTestUtils::fillByteVectorAsTyped<int>(byteInputData);
    }
    auto res1 = CalculateRefs(f1, byteInputData);
    auto res2 = CalculateRefs(f2, byteInputData);

    auto vec1 = FuncTestUtils::getTypedVector<float>(res1);
    auto vec2 = FuncTestUtils::getTypedVector<float>(res2);

    // get real typed vectors
    compareTypedVectors<float>(vec1, vec2);
//    CompareBytes(res1, res2, InferenceEngine::Precision::FP32);
    return true;
}