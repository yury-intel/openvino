// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_concat_memory.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"

namespace LayerTestsDefinitions {

using namespace CommonTestUtils;
using namespace InferenceEngine;

std::string SplitConcatMemory::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitConcatMemory::SetUp() {
    SizeVector shape;
    std::tie(inPrc, shape, targetDevice) = this->GetParam();

    axis = 1;

    auto shape_14 = shape;
    shape_14[axis] /= 4;
    auto shape_34 = shape;
    shape_34[axis] -= shape_14[axis];

    /*
     *    Cyclic buffer length of 4
     *        ______   ______
     *       [_mem1_] [_inp1_]
     *          _|______|_
     *         [_cocncat__]
     *         _____|______
     *      __|___      ___|__
     *     [_pow1_]    [_spl1_]
     *        |         |    |
     *      __|___         __|___
     *     [_out1_]       [_mem2_]
     */
    IRBuilder_v7 builder("SplitConcatMemoryPattern");
    auto inp1 = builder.AddLayer("inp1", "Input" )
            .AddOutPort(inPrc, shape_14)
            .getLayer();
    auto mem1 = builder.AddLayer("mem1", "Memory", {
                 {"id",  "holder_1"},
                 {"index", "1"},
                 {"size", "2"}
            })
            .AddOutPort(inPrc, shape_34)
            .getLayer();
    auto cnc1 = builder.AddLayer("cnc1", "Concat")
            .AddInPort(inPrc, shape_34)
            .AddInPort(inPrc, shape_14)
            .AddOutPort(inPrc, shape)
            .getLayer();
    auto spl1 = builder.AddLayer("spl1", "Split" )
            .AddInPort(inPrc, shape)
            .AddOutPort(inPrc, shape_14)
            .AddOutPort(inPrc, shape_34)
            .getLayer();
    auto pow1 = builder.AddLayer("pow1", "Power", {
                {"shift", "1"},
                {"scale", "1"},
                {"power", "1"},
            })
            .AddInPort(inPrc, shape)
            .AddOutPort(inPrc, shape)
            .getLayer();
    auto mem2 = builder.AddLayer("mem2", "Memory", {
                {"id",  "holder_1"},
                {"index", "0"},
                {"size", "2"}
            })
            .AddInPort(inPrc, shape_34)
            .getLayer();

    builder.AddEdge(inp1.out(0), cnc1.in(1))
           .AddEdge(mem1.out(0), cnc1.in(0))
           .AddEdge(cnc1.out(0), spl1.in(0))
           .AddEdge(cnc1.out(0), pow1.in(0))
           .AddEdge(spl1.out(1), mem2.in(0));

    net_xml = builder.serialize();
    net_bin = make_plain_blob(Precision::U8, SizeVector{});
    net_bin->allocate();
}

TEST_P(SplitConcatMemory, ciclicBufferCorrectness) {
    auto ie = InferenceEngine::Core();
    auto net = ie.ReadNetwork(net_xml, net_bin);
    auto exe_net = ie.LoadNetwork(net, "CPU");
    auto inf_reg = exe_net.CreateInferRequest();

    /*
     * cnc1 out  |  mem      | In|
     *           |===============|
     * iter_1    | 0 | 0 | 0 | 1 |
     * iter_2    | 0 | 0 | 1 | 2 |
     * iter 3    | 0 | 1 | 2 | 3 |
     */

    auto i_blob = inf_reg.GetBlob("inp1");
    auto o_blob = inf_reg.GetBlob("pow1");

    auto o_blob_ref = make_blob_with_precision(o_blob->getTensorDesc());
    o_blob_ref->allocate();

    auto fill_by_quarter = [this] (Blob::Ptr& blob, std::vector<float> vals) {
        IE_ASSERT(vals.size() == 4);
        auto quarter_blocked_shape = blob->getTensorDesc().getDims();

        // splis axis dimension into chunk
        IE_ASSERT(quarter_blocked_shape[axis] % vals.size() == 0);
        quarter_blocked_shape[axis] /= vals.size();
        quarter_blocked_shape.insert(quarter_blocked_shape.begin() + axis, vals.size());

        auto quarter_blocked_view = make_reshape_view(blob, quarter_blocked_shape);
        fill_data_with_broadcast(quarter_blocked_view, axis, vals);
    };

    // iteration 1
    fill_data_const(i_blob, 1);
    fill_by_quarter(o_blob_ref, {1, 1, 1, 2});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);

    // iteration 2
    fill_data_const(i_blob, 2);
    fill_by_quarter(o_blob_ref, {1, 1, 2, 3});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);

    // iteration 3
    fill_data_const(i_blob, 3);
    fill_by_quarter(o_blob_ref, {1, 2, 3, 4});
    inf_reg.Infer();
    Compare(o_blob_ref, o_blob);
}

}  // namespace LayerTestsDefinitions