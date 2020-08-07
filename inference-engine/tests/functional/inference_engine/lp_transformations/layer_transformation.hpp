// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional_test_utils/blob_utils.hpp>
#include "common_test_utils/test_common.hpp"
#include "transformations/low_precision/layer_transformation.hpp"
#include "transformations/low_precision/transformation_context.hpp"
#include "transformations/low_precision/transformer.hpp"

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

class LayerTransformation : public FuncTestUtils::ComparableNGTestCommon {
    void Infer() override {}

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
    bool compareResults(std::shared_ptr<ngraph::Function> f1, std::shared_ptr<ngraph::Function> f2);

protected:
    void transform(std::shared_ptr<ngraph::Function> function);
    void transform(
        std::shared_ptr<ngraph::Function> function,
        std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr>& transformations);

    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};
