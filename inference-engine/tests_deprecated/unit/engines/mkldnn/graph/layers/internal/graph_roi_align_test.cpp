// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <ie_core.hpp>
#include <ie_system_conf.h>

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct roi_align_test_params {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in1;

    struct {
        size_t n;
        size_t c;
    } in2;

    size_t pooled_h;
    size_t pooled_w;
    std::string method;
    float spatial_scale;
    size_t sampling_ratio;

    size_t num_prim_desc;

    int selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_roialign(const InferenceEngine::TBlob<data_t> &src, const InferenceEngine::TBlob<data_t> &roi,
                  InferenceEngine::TBlob<data_t> &dst_blob, roi_align_test_params& params) {
    data_t* dst = dst_blob.data();
    const data_t* src_data = src.readOnly();
    const data_t* src_roi = roi.readOnly();

    int C = src.getTensorDesc().getDims()[1];
    int H = src.getTensorDesc().getDims()[2];
    int W = src.getTensorDesc().getDims()[3];

    int ROIS = roi.getTensorDesc().getDims()[0];

    double spatial_scale = params.spatial_scale;
    int pooled_h = params.pooled_h;
    int pooled_w = params.pooled_w;

    auto *arg_max_ = new data_t[dst_blob.size()];

    for (size_t i = 0; i < dst_blob.size(); i++) {
        arg_max_[i] = -1;
        dst[i] = -FLT_MAX;
    }

    int roi_off;

    for (int n = 0; n < ROIS; ++n) {
        if(roi.getTensorDesc().getDims().size() == 4) {
            roi_off = n*roi.getTensorDesc().getDims()[1]*roi.getTensorDesc().getDims()[2]*roi.getTensorDesc().getDims()[3];
        }
        else {
            roi_off = n*roi.getTensorDesc().getDims()[1];
        }

        const data_t* src_roi_ptr = &src_roi[roi_off];

        int roi_batch_ind = src_roi_ptr[0];
        int roi_start_w = round(src_roi_ptr[1] * spatial_scale);
        int roi_start_h = round(src_roi_ptr[2] * spatial_scale);
        int roi_end_w = round(src_roi_ptr[3] * spatial_scale);
        int roi_end_h = round(src_roi_ptr[4] * spatial_scale);

        int roi_height = (std::max)(roi_end_h - roi_start_h + 1, 1);
        int roi_width = (std::max)(roi_end_w - roi_start_w + 1, 1);

        for (int c = 0; c < C; ++c) {

            for (int ph = 0; ph < pooled_h; ++ph) {
                for (int pw = 0; pw < pooled_w; ++pw) {
                    int hstart = (ph * roi_height) / pooled_h;
                    if ( (hstart * pooled_h) > (ph * roi_height) ) {
                        --hstart;
                    }

                    int wstart = (pw * roi_width) / pooled_w;
                    if ( (wstart * pooled_w) > (pw * roi_width) ) {
                        --wstart;
                    }

                    int hend = ((ph + 1) * roi_height) / pooled_h;
                    if ( (hend * pooled_h) < ((ph + 1) * roi_height) ) {
                        ++hend;
                    }

                    int wend = ((pw + 1) * roi_width) / pooled_w;
                    if ( (wend * pooled_w) < ((pw + 1) * roi_width) ) {
                        ++wend;
                    }

                    hstart = (std::min)((std::max)(hstart + roi_start_h, 0), H);
                    hend = (std::min)((std::max)(hend + roi_start_h, 0), H);
                    wstart = (std::min)((std::max)(wstart + roi_start_w, 0), W);
                    wend = (std::min)((std::max)(wend + roi_start_w, 0), W);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);

                    const int pool_index = n*dst_blob.getTensorDesc().getDims()[3]*dst_blob.getTensorDesc().getDims()[2]*dst_blob.getTensorDesc().getDims()[1] +
                                           c*dst_blob.getTensorDesc().getDims()[3]*dst_blob.getTensorDesc().getDims()[2] + ph*dst_blob.getTensorDesc().getDims()[3] + pw;

                    if (is_empty) {
                        dst[pool_index] = 0;
                        arg_max_[pool_index] = -1;
                    }

                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            int src_index_data = roi_batch_ind*src.getTensorDesc().getDims()[1]*src.getTensorDesc().getDims()[2]*src.getTensorDesc().getDims()[3] +
                                                 c*src.getTensorDesc().getDims()[2]*src.getTensorDesc().getDims()[3] + h*src.getTensorDesc().getDims()[3] + w;
                            data_t batch_data = src_data[src_index_data];

                            if (batch_data > dst[pool_index]) {
                                dst[pool_index] = batch_data;
                                arg_max_[pool_index] = batch_data;
                            }
                        }
                    }
                }
            }
        }
    }
    delete[] arg_max_;
}

template <typename data_t>
void ref_roialign1(const InferenceEngine::TBlob<data_t> &src, const InferenceEngine::TBlob<data_t> &roi,
                   InferenceEngine::TBlob<data_t> &dst_blob, roi_align_test_params& params) {
    data_t* dst = dst_blob.data();
    const data_t* src_data = src.readOnly();
    const data_t* src_roi = roi.readOnly();

    int C = src.getTensorDesc().getDims()[1];
    int H = src.getTensorDesc().getDims()[2];
    int W = src.getTensorDesc().getDims()[3];

    int ROIS = roi.getTensorDesc().getDims()[0];

    double spatial_scale = params.spatial_scale;
    int sampling_ratio = params.sampling_ratio;
    int pooled_h = params.pooled_h;
    int pooled_w = params.pooled_w;

    auto mode = params.method;

    auto *arg_max_ = new data_t[dst_blob.size()];

    for (size_t i = 0; i < dst_blob.size(); i++) {
        arg_max_[i] = -1;
        dst[i] = -FLT_MAX;
    }

    int roi_off;
// TODO - real rois
    for (int n = 0; n < ROIS; ++n) {
        // TODO - check boundaries of batch index

        roi_off = n * 5;  // if batch id contained in same input
        const data_t* src_roi_ptr = &src_roi[roi_off];
// TODO: remove to another input
        int roi_batch_ind = src_roi_ptr[0];
        data_t x1 = src_roi_ptr[1] * spatial_scale;
        data_t y1 = src_roi_ptr[2] * spatial_scale;
        data_t x2 = src_roi_ptr[3] * spatial_scale;
        data_t y2 = src_roi_ptr[4] * spatial_scale;

        data_t roi_height = std::max(y2 - y1, 1.0f);
        data_t roi_width = std::max(x2 - x1, 1.0f);
        data_t bin_height = roi_height / pooled_h;
        data_t bin_width = roi_width / pooled_w;

        auto sampling_ratio_x = sampling_ratio == 0 ? static_cast<int>(ceil(bin_width)) : sampling_ratio;
        auto sampling_ratio_y = sampling_ratio == 0 ? static_cast<int>(ceil(bin_height)) : sampling_ratio;

        uint64_t num_samples_in_bin = sampling_ratio_x * sampling_ratio_y;

        data_t sample_distance_x = bin_width / sampling_ratio_x;
        data_t sample_distance_y = bin_height / sampling_ratio_y;
        // prepare arrays for sampling points and weights
        std::vector<std::pair<int, int>> point_vector;
        std::vector<data_t> weight_vector;
        point_vector.reserve(4 * num_samples_in_bin * pooled_h * pooled_w);
        weight_vector.reserve(4 * num_samples_in_bin * pooled_h * pooled_w);

// iterate over bins
        for (int y_bin_ind = 0; y_bin_ind < pooled_h; ++y_bin_ind) {
            for (int x_bin_ind = 0; x_bin_ind < pooled_w; ++x_bin_ind) {
                // run into bin
                for (unsigned int y_sample_ind = 0; y_sample_ind < sampling_ratio_y;
                     y_sample_ind++) {
                    data_t sample_y = y1 + y_bin_ind * bin_height +
                                      sample_distance_y * (0.5f + y_sample_ind);
                    for (int64_t x_sample_ind = 0; x_sample_ind < sampling_ratio_x;
                         x_sample_ind++) {
                        data_t sample_x = x1 + x_bin_ind * bin_width +
                                          sample_distance_x * (0.5f + x_sample_ind);

                        if (sample_x < -1.0 || sample_x > W ||
                            sample_y < -1.0 || sample_y > H)
                        {
                            // For this sample we save 4x point (0,0) with weight 0
                            point_vector.insert(point_vector.end(), 4, {0, 0});
                            weight_vector.insert(weight_vector.end(), 4, data_t{0});
                            continue;
                        }
                        sample_x = std::max(sample_x, data_t{0});
                        sample_y = std::max(sample_y, data_t{0});

                        auto sample_y_low = static_cast<unsigned int>(sample_y);
                        auto sample_x_low = static_cast<unsigned int>(sample_x);
                        unsigned int sample_y_high;
                        unsigned int sample_x_high;
                        if (sample_y_low >= H - 1)
                        {
                            sample_y_high = sample_y_low = H - 1;
                            sample_y = static_cast<data_t>(sample_y_low);
                        }
                        else
                        {
                            sample_y_high = sample_y_low + 1;
                        }
                        if (sample_x_low >= H - 1)
                        {
                            sample_x_high = sample_x_low = W - 1;
                            sample_x = static_cast<data_t>(sample_x_low);
                        }
                        else
                        {
                            sample_x_high = sample_x_low + 1;
                        }
                        point_vector.push_back({sample_y_low, sample_x_low});
                        point_vector.push_back({sample_y_low, sample_x_high});
                        point_vector.push_back({sample_y_high, sample_x_low});
                        point_vector.push_back({sample_y_high, sample_x_high});

                        // weight calculation for bilinear interpolation
                        auto ly = sample_y - sample_y_low;
                        auto lx = sample_x - sample_x_low;
                        auto hy = 1.0f - ly;
                        auto hx = 1.0f - lx;

                        weight_vector.push_back(hy * hx);
                        weight_vector.push_back(hy * lx);
                        weight_vector.push_back(ly * hx);
                        weight_vector.push_back(ly * lx);
                    }
                }
            }
        }

        for (int c = 0; c < C; ++c) {  // parallel for
            unsigned int sample_index = 0;
            for (int y_bin_ind = 0; y_bin_ind < pooled_h; ++y_bin_ind) {
                for (int x_bin_ind = 0; x_bin_ind < pooled_w; ++x_bin_ind) {
                    // run into bin
                    data_t pooled_value = 0;
                    for (unsigned int bin_sample_ind = 0;
                         bin_sample_ind < num_samples_in_bin;
                         bin_sample_ind++) {
                        int bin_offset = roi_batch_ind * C * H * W + c * H * W;
                        int part1_index = bin_offset + point_vector[sample_index].first * W +
                                          point_vector[sample_index].second;
                        data_t part1 = src_data[part1_index];
                        int part2_index = bin_offset + point_vector[sample_index + 1].first * W +
                                          point_vector[sample_index + 1].second;
                        data_t part2 = src_data[part2_index];
                        int part3_index = bin_offset + point_vector[sample_index + 2].first * W +
                                          point_vector[sample_index + 2].second;
                        data_t part3 = src_data[part3_index];
                        int part4_index = bin_offset + point_vector[sample_index + 3].first * W +
                                          point_vector[sample_index + 3].second;
                        data_t part4 = src_data[part4_index];


                        // TODO: replace to enum
                        switch (1)
                        {
                            case 1:
                            {
                                data_t sample_value = std::max(
                                        {weight_vector[sample_index] * part1,
                                         weight_vector[sample_index + 1] * part2,
                                         weight_vector[sample_index + 2] * part3,
                                         weight_vector[sample_index + 3] * part4});

                                pooled_value = sample_value > pooled_value ? sample_value
                                                                           : pooled_value;
                                break;
                            }
                            case 2:
                            default:
                            {
                                data_t sample_value =
                                        weight_vector[sample_index] * part1 +
                                        weight_vector[sample_index + 1] * part2 +
                                        weight_vector[sample_index + 2] * part3 +
                                        weight_vector[sample_index + 3] * part4;
                                pooled_value += sample_value / (num_samples_in_bin);
                            }
                        }
                        sample_index += 4;
                    }
                    // write pooled value
                    size_t dst_index = n * C * pooled_h * pooled_w + c * pooled_h * pooled_w +
                                       y_bin_ind * pooled_w + x_bin_ind;
                    dst[dst_index] = pooled_value;
                }
            }
        }
    }
    delete[] arg_max_;
}

class MKLDNNGraphRoiAlignTests: public TestsCommon,
                                public WithParamInterface<roi_align_test_params> {
    std::string model_t = R"V0G0N(
<Net Name="ROIAlign_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN1_</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>_IN2_</dim>
                    <dim>_IC2_</dim>
                </port>
            </output>
        </layer>
        <layer name="roi_align" id="2" type="ROIAlign" precision="FP32">
            <data pooled_h="_PH_" pooled_w="_PW_" method="_MTH_" spatial_scale="_SS_" sampling_ratio="_SR_"/>
            <input>
                <port id="2">
                    <dim>_IN1_</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
                <port id="3">
                    <dim>_IN2_</dim>
                    <dim>_IC2_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_ON_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(roi_align_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW1_", p.in1.w);
        REPLACE_WITH_NUM(model, "_IH1_", p.in1.h);
        REPLACE_WITH_NUM(model, "_IC1_", p.in1.c);
        REPLACE_WITH_NUM(model, "_IN1_", p.in1.n);

        REPLACE_WITH_NUM(model, "_IC2_", p.in2.c);
        REPLACE_WITH_NUM(model, "_IN2_", p.in2.n);

        REPLACE_WITH_NUM(model, "_OW_", p.pooled_w);
        REPLACE_WITH_NUM(model, "_OH_", p.pooled_h);
        REPLACE_WITH_NUM(model, "_OC_", (std::max)(p.in1.c, p.in2.c));
        REPLACE_WITH_NUM(model, "_ON_", (std::max)(p.in1.n, p.in2.n));

        REPLACE_WITH_NUM(model, "_PH_", p.pooled_h);
        REPLACE_WITH_NUM(model, "_PW_", p.pooled_w);
        REPLACE_WITH_NUM(model, "_MTH_", p.method);
        REPLACE_WITH_NUM(model, "_SS_", p.spatial_scale);
        REPLACE_WITH_NUM(model, "_SR_", p.sampling_ratio);


        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            roi_align_test_params p = ::testing::WithParamInterface<roi_align_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::ROIAlign) {
                    ASSERT_LE(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            InferenceEngine::SizeVector dims_src = {p.in1.n, p.in1.c, p.in1.h, p.in1.w};

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::SizeVector dims_roi = {p.in2.n, p.in2.c};

            InferenceEngine::Blob::Ptr roi = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_roi, InferenceEngine::NC});
            roi->allocate();
            fill_roi_data(roi->buffer(), roi->size());

            InferenceEngine::TBlob<float>* roiPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(roi.get());

            if (roiPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", roi));

            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            ref_roialign1(*srcPtr, *roiPtr, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }

    static void fill_roi_data(float *data, size_t size, size_t duty_ratio = 10) {
        for (size_t i = 0; i < size; i++) {
            if (i % 5 == 0) {
                data[i] = 0.0f;
            } else if (i % 5 == 1 || i % 5 == 3) {
                data[i] = 1.0f;
            } else if (i % 5 == 2 || i % 5 == 4) {
                data[i] = 3.0f;
            }
        }
    }
};

TEST_P(MKLDNNGraphRoiAlignTests, TestsRoiAlign) {}

const size_t expect_num_impl = 1;

INSTANTIATE_TEST_CASE_P(
        TestsRoiAlign, MKLDNNGraphRoiAlignTests,
        ::testing::Values(
                roi_align_test_params{
                        {1, 256, 39, 64},  // in1
                        {150, 5},          // in2
                        6, 6,              // pool H and W
                        "max",             // pooling method
                        1.0f,           // spatial_scale
                        2,              // sampling ratio
                        expect_num_impl,   // num_prim_desc (platform dependent)
                        MKLDNNPlugin::impl_desc_type::unknown
                }));
