// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <fstream>

#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"

namespace LayerTestsDefinitions {
using namespace CommonTestUtils;
using namespace InferenceEngine;

class MemoryConv : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                   public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
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

protected:
    void SetUp() {
        SizeVector shape;
        std::tie(inPrc, shape, targetDevice) = this->GetParam();

        net_bin_v6 = make_plain_blob(Precision::FP32, SizeVector{(200 * 200 + 200 + 200 * 200 + 200) * sizeof(float)});
        net_bin_v6->allocate();
    }
    std::string net_xml_v6 = R"V0G0N(
<net name="model" version="6">
    <layers>
        <layer id="0" name="Memory_1" precision="FP32" type="Memory">
            <data id="r_1-3" index="1" size="2" />
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Input_2" precision="FP32" type="input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Eltwise_3" precision="FP32" type="Eltwise">
            <data operation="mul" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="Activation_3_2" precision="FP32" type="Activation">
            <data type="sigmoid" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="Fc_4" precision="FP32" type="InnerProduct" >
            <data out-size="200" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
            <weights offset="0" size="160000" />
            <biases offset="160000" size="800" />
        </layer>
        <layer id="5" name="Memory_5" precision="FP32" type="Memory">
            <data id="r_1-3" index="0" size="2" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
        </layer>
       <layer id="6" name="Fc_6" precision="FP32" type="InnerProduct"  >
            <data out-size="200" />
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>200</dim>
                </port>
            </output>
            <weights offset="160800" size="160000" />
            <biases offset="320800" size="800" />
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
        <edge from-layer="3" from-port="1" to-layer="4" to-port="0" />
        <edge from-layer="4" from-port="1" to-layer="5" to-port="0" />
        <edge from-layer="4" from-port="1" to-layer="6" to-port="0" />
    </edges>
</net>
)V0G0N";
    InferenceEngine::Blob::Ptr net_bin_v6;
};

TEST_P(MemoryConv, inferV6Model) {
    auto ie = InferenceEngine::Core();
    auto net = ie.ReadNetwork(net_xml_v6, net_bin_v6);
    auto exe_net = ie.LoadNetwork(net, "CPU");
    auto inf_reg = exe_net.CreateInferRequest();

    CNNNetwork info = exe_net.GetExecGraphInfo();

    auto cc = inf_reg.GetPerformanceCounts();
    std::string mem0_str = "Memory_1";
    std::string mem1_str = "Memory_5";
    std::string exp_prec_str = "BF16";

    auto mem0 = std::find_if(cc.begin(), cc.end(), [mem0_str] (const std::pair<std::string, InferenceEngineProfileInfo> & c) -> bool {
        return c.first.substr(0, mem0_str.length()) == mem0_str;;
    });
    auto mem1 = std::find_if(cc.begin(), cc.end(), [mem1_str] (const std::pair<std::string, InferenceEngineProfileInfo> & c) -> bool {
        return c.first.substr(0, mem1_str.length()) == mem1_str;;
    });
    std::string execType0 = mem0->second.exec_type;
    std::string pfPrecision0 = execType0.substr(execType0.length() - exp_prec_str.length(), exp_prec_str.length());
    std::string execType1 = mem1->second.exec_type;
    std::string pfPrecision1 = execType1.substr(execType1.length() - exp_prec_str.length(), exp_prec_str.length());
    ASSERT_EQ(pfPrecision0, exp_prec_str);
    ASSERT_EQ(pfPrecision1, exp_prec_str);
}
}  // namespace LayerTestsDefinitions

namespace {
using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::SizeVector> shapes = {
        {3, 8},
};

INSTANTIATE_TEST_CASE_P(CPU, MemoryConv,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapes),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MemoryConv::getTestCaseName);
}  // namespace