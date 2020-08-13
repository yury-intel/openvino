// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "transformations/low_precision/layer_transformation.hpp"
#include "transformations/low_precision/transformation_context.hpp"
#include "transformations/low_precision/transformer.hpp"
#include <functional_test_utils/blob_utils.hpp>
typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

class LayerTransformation : public CommonTestUtils::TestsCommon {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8U8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8AndI8();

    static std::string toString(const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type& type,
        const ngraph::Shape& shape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);

protected:
    void transform(std::shared_ptr<ngraph::Function> function);
    void transform(
        std::shared_ptr<ngraph::Function> function,
        std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr>& transformations);
    template<class T>
    void compareTypedVectors(const std::vector<std::vector<T>>& v1, const std::vector<std::vector<T>>& v2);
    template<class T>
    void compareBytes(const std::vector<std::vector<std::uint8_t>>& expectedVector, const std::vector<std::vector<std::uint8_t>>& actualVector);
    template<class T>
    void compareValues(const T *expected, const T *actual, std::size_t size, T thr);
    template<class T>
    void compareValues(const void *expected, const void *actual, std::size_t size);
    virtual std::vector<std::vector<std::uint8_t>> CalculateRefs(std::shared_ptr<ngraph::Function> _function, std::vector<std::vector<std::uint8_t>> _inputs);
    bool compareResults(std::shared_ptr<ngraph::Function> f1, std::shared_ptr<ngraph::Function> f2);

    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
    std::vector<std::vector<std::uint8_t>> byteInputData;
    std::vector<std::vector<std::uint8_t>> actualByteOutput;
    std::vector<std::vector<std::uint8_t>> expectedByteOutput;
    float threshold = 1e-2f;
};
