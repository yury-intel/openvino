// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_roi_align_node.h"
#include "desc_iterator.hpp"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include "jit_generator.hpp"
#include "ie_parallel.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

MKLDNNROIAlignNode::MKLDNNROIAlignNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
                                       MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNROIAlignNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    class CNNLayer *genericLayer = getCnnLayer().get();
    if (genericLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert ROIPooling layer.";

    std::string errorPrefix = "ROIPooling layer with name '" + getName() + "' ";

    if (getParentEdges().size() != 2)
        THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 0th input with rank: " << getParentEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims().ndims() != 2) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getChildEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support output with rank: " << getChildEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims()[1] != 5) {  // TODO: change
        THROW_IE_EXCEPTION << errorPrefix << "has invalid shape on 1st input: ["
                           << getParentEdgeAt(1)->getDims()[0] << "," << getParentEdgeAt(1)->getDims()[1] << "]";
    }

    pooled_h = genericLayer->GetParamAsInt("pooled_h");
    pooled_w = genericLayer->GetParamAsInt("pooled_w");
    spatial_scale = genericLayer->GetParamAsFloat("spatial_scale");
    sampling_ratio = genericLayer->GetParamAsInt("sampling_ratio");
    std::string m = genericLayer->GetParamAsString("method", "max");
    if (m == "max") {
        opType = ROIAlignOpType::Max;
    } else if (m == "bilinear") {  // TODO: remove
        opType = ROIAlignOpType::Bilinear;
    } else if (m == "avg") {
        opType = ROIAlignOpType::Avg;
    } else {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support roi pooling method: " << m;
    }
}

void MKLDNNROIAlignNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.inConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.inConfs[1].constant = false;
    config.inConfs[1].inPlace = -1;

    config.outConfs.resize(1);
    config.outConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;

//    auto parentDims = getParentEdgeAt(0)->getDims();
//    auto format = mayiuse(avx512_common) ? memory::format::nChw16c : memory::format::nChw8c;
//    impl_desc_type impl_type;
//    if (mayiuse(cpu::avx512_common)) {
//        impl_type = impl_desc_type::jit_avx512;
//    } else if (mayiuse(cpu::avx2)) {
//        impl_type = impl_desc_type::jit_avx2;
//    } else if (mayiuse(cpu::sse42)) {
//        impl_type = impl_desc_type::jit_sse42;
//    } else {
//        impl_type = impl_desc_type::ref;
//    }
//    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::f32, format);
//    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
//    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::f32, format);
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::f32, memory::nchw);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::f32, memory::nchw);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nchw});
}

void MKLDNNROIAlignNode::execute(mkldnn::stream strm) {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

//    const auto *src_data = reinterpret_cast<const float *>(srcMemory0.GetData()) + srcMemory0.GetDescriptor().data.layout_desc.blocking.offset_padding;
//    const auto *src_roi = reinterpret_cast<const float *>(srcMemory1.GetData()) + srcMemory1.GetDescriptor().data.layout_desc.blocking.offset_padding;
//    float *dst = reinterpret_cast<float *>(dstMemory.GetData()) + dstMemory.GetDescriptor().data.layout_desc.blocking.offset_padding;

    const auto *src_data = reinterpret_cast<const float *>(srcMemory0.GetData());
    const auto *src_roi = reinterpret_cast<const float *>(srcMemory1.GetData());
    float *dst = reinterpret_cast<float *>(dstMemory.GetData());

    auto config = getSelectedPrimitiveDescriptor()->getConfig();
//
//    auto src_strides = config.inConfs[0].desc.getBlockingDesc().getStrides();
//    auto dst_strides = config.outConfs[0].desc.getBlockingDesc().getStrides();

//    size_t src_roi_step = config.inConfs[1].desc.getBlockingDesc().getStrides()[0];
    auto nominalRoiCount = static_cast<int>(srcMemory1.GetDims()[0]);
    int real_rois = 0;
    int C = static_cast<int>(srcMemory0.GetDims()[1]);
    int H = static_cast<int>(srcMemory0.GetDims()[2]);
    int W = static_cast<int>(srcMemory0.GetDims()[3]);
    auto mode = opType;

    for (; real_rois < nominalRoiCount; real_rois++) {
//        size_t roi_off = real_rois * src_roi_step;
        const float *src_roi_ptr = &src_roi[real_rois];
        int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);
        if (roi_batch_ind == -1) {
            break;
        }
    }

    int roi_off;

    for (int n = 0; n < real_rois; ++n) {
        // TODO - check boundaries of batch index

        roi_off = n * 5;  // if batch id contained in same input
        const float* src_roi_ptr = &src_roi[roi_off];
// TODO: remove to another input
        int roi_batch_ind = src_roi_ptr[0];
        float x1 = src_roi_ptr[1] * spatial_scale;
        float y1 = src_roi_ptr[2] * spatial_scale;
        float x2 = src_roi_ptr[3] * spatial_scale;
        float y2 = src_roi_ptr[4] * spatial_scale;

        float roi_height = std::max(y2 - y1, 1.0f);
        float roi_width = std::max(x2 - x1, 1.0f);
        float bin_height = roi_height / pooled_h;
        float bin_width = roi_width / pooled_w;

        auto sampling_ratio_x = sampling_ratio == 0 ? static_cast<int>(ceil(bin_width)) : sampling_ratio;
        auto sampling_ratio_y = sampling_ratio == 0 ? static_cast<int>(ceil(bin_height)) : sampling_ratio;

        uint64_t num_samples_in_bin = sampling_ratio_x * sampling_ratio_y;

        float sample_distance_x = bin_width / sampling_ratio_x;
        float sample_distance_y = bin_height / sampling_ratio_y;
        // prepare arrays for sampling points and weights
        std::vector<std::pair<int, int>> point_vector;
        std::vector<float> weight_vector;
        point_vector.reserve(4 * num_samples_in_bin * pooled_h * pooled_w);
        weight_vector.reserve(4 * num_samples_in_bin * pooled_h * pooled_w);

// iterate over bins
        for (int y_bin_ind = 0; y_bin_ind < pooled_h; ++y_bin_ind) {
            for (int x_bin_ind = 0; x_bin_ind < pooled_w; ++x_bin_ind) {
                // run into bin
                for (unsigned int y_sample_ind = 0; y_sample_ind < sampling_ratio_y;
                     y_sample_ind++) {
                    float sample_y = y1 + y_bin_ind * bin_height +
                                     sample_distance_y * (0.5f + y_sample_ind);
                    for (int64_t x_sample_ind = 0; x_sample_ind < sampling_ratio_x;
                         x_sample_ind++) {
                        float sample_x = x1 + x_bin_ind * bin_width +
                                         sample_distance_x * (0.5f + x_sample_ind);

                        if (sample_x < -1.0 || sample_x > W ||
                            sample_y < -1.0 || sample_y > H)
                        {
                            // For this sample we save 4x point (0,0) with weight 0
                            point_vector.insert(point_vector.end(), 4, {0, 0});
                            weight_vector.insert(weight_vector.end(), 4, float{0});
                            continue;
                        }
                        sample_x = std::max(sample_x, float{0});
                        sample_y = std::max(sample_y, float{0});

                        auto sample_y_low = static_cast<unsigned int>(sample_y);
                        auto sample_x_low = static_cast<unsigned int>(sample_x);
                        unsigned int sample_y_high;
                        unsigned int sample_x_high;
                        if (sample_y_low >= H - 1)
                        {
                            sample_y_high = sample_y_low = H - 1;
                            sample_y = static_cast<float>(sample_y_low);
                        }
                        else
                        {
                            sample_y_high = sample_y_low + 1;
                        }
                        if (sample_x_low >= H - 1)
                        {
                            sample_x_high = sample_x_low = W - 1;
                            sample_x = static_cast<float>(sample_x_low);
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
                    float pooled_value = 0;
                    for (unsigned int bin_sample_ind = 0;
                         bin_sample_ind < num_samples_in_bin;
                         bin_sample_ind++) {
                        int bin_offset = roi_batch_ind * C * H * W + c * H * W;
                        int part1_index = bin_offset + point_vector[sample_index].first * W +
                                          point_vector[sample_index].second;
                        float part1 = src_data[part1_index];
                        int part2_index = bin_offset + point_vector[sample_index + 1].first * W +
                                          point_vector[sample_index + 1].second;
                        float part2 = src_data[part2_index];
                        int part3_index = bin_offset + point_vector[sample_index + 2].first * W +
                                          point_vector[sample_index + 2].second;
                        float part3 = src_data[part3_index];
                        int part4_index = bin_offset + point_vector[sample_index + 3].first * W +
                                          point_vector[sample_index + 3].second;
                        float part4 = src_data[part4_index];


                        // TODO: replace to enum
                        switch (1)
                        {
                            case 1:
                            {
                                float sample_value = std::max(
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
                                float sample_value =
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

//    parallel_for4d(MB, cb_work, jpp.oh, jpp.ow, [&](int n, int cbb, int oh, int ow) {
//        auto arg = jit_roi_pooling_call_args();
//
//        int cb = cbb * jpp.nb_c_blocking;
//        int cb_num = jpp.nb_c_blocking;
//        int c_block = jpp.c_block;
//
//        arg.c_blocks = std::min(cb + cb_num, jpp.nb_c) - cb;
//
//        if (n >= real_rois) {
//            if (roi_pooling_kernel) {
//                arg.bin_area = 0;
//                arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
//            } else {
//                for (int c = 0; c < c_block; c++) {
//                    dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] = 0;
//                }
//            }
//
//            (*roi_pooling_kernel)(&arg);
//        } else {
//            size_t roi_off = n * src_roi_step;
//            const float* src_roi_ptr = &src_roi[roi_off];
//
//            int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);
//
//            if (jpp.alg == ROIAlignOpType::Max) {
//                int roi_start_w = static_cast<int>(round(src_roi_ptr[1] * jpp.spatial_scale));
//                int roi_start_h = static_cast<int>(round(src_roi_ptr[2] * jpp.spatial_scale));
//                int roi_end_w = static_cast<int>(round(src_roi_ptr[3] * jpp.spatial_scale));
//                int roi_end_h = static_cast<int>(round(src_roi_ptr[4] * jpp.spatial_scale));
//
//                int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
//                int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
//
//
//                int hstart = (oh * roi_height) / jpp.pooled_h;
//                if ((hstart * jpp.pooled_h) > (oh * roi_height)) {
//                    --hstart;
//                }
//
//                int wstart = (ow * roi_width) / jpp.pooled_w;
//                if ((wstart * jpp.pooled_w) > (ow * roi_width)) {
//                    --wstart;
//                }
//
//                int hend = ((oh + 1) * roi_height) / jpp.pooled_h;
//                if ((hend * jpp.pooled_h) < ((oh + 1) * roi_height)) {
//                    ++hend;
//                }
//
//                int wend = ((ow + 1) * roi_width) / jpp.pooled_w;
//                if ((wend * jpp.pooled_w) < ((ow + 1) * roi_width)) {
//                    ++wend;
//                }
//
//                hstart = std::min(std::max(hstart + roi_start_h, 0), jpp.ih);
//                hend = std::min(std::max(hend + roi_start_h, 0), jpp.ih);
//                wstart = std::min(std::max(wstart + roi_start_w, 0), jpp.iw);
//                wend = std::min(std::max(wend + roi_start_w, 0), jpp.iw);
//
//                if (roi_pooling_kernel) {
//                    arg.src = &src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] + hstart * src_strides[2] + wstart * src_strides[3]];
//                    arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
//
//                    arg.bin_area = (hend - hstart) * (wend - wstart);
//                    arg.kh = hend - hstart;
//                    arg.kw = wend - wstart;
//                } else {
//                    for (int c = 0; c < c_block; c++) {
//                        const size_t pool_index = n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c;
//                        if ((hend <= hstart) || (wend <= wstart)) {
//                            dst[pool_index] = 0;
//                        } else {
//                            for (int h = hstart; h < hend; ++h) {
//                                for (int w = wstart; w < wend; ++w) {
//                                    float batch_data = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
//                                                                h * src_strides[2] + w * src_strides[3] + c];
//
//                                    if (batch_data > dst[pool_index]) {
//                                        dst[pool_index] = batch_data;
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//            } else {
//                float roi_start_w_ = src_roi_ptr[1];
//                float roi_start_h_ = src_roi_ptr[2];
//                float roi_end_w_   = src_roi_ptr[3];
//                float roi_end_h_   = src_roi_ptr[4];
//
//                float height_scale = ((roi_end_h_ - roi_start_h_) * (jpp.ih - 1)) / (jpp.pooled_h - 1);
//                float width_scale  = ((roi_end_w_ - roi_start_w_) * (jpp.iw - 1)) / (jpp.pooled_w - 1);
//
//                float in_y = (oh * height_scale + roi_start_h_ * (jpp.ih - 1));
//                float in_x = (ow * width_scale  + roi_start_w_ * (jpp.iw - 1));
//
//                if (in_y < 0 || in_y > jpp.ih - 1 || in_x < 0 || in_x > jpp.iw - 1) {
//                    if (roi_pooling_kernel) {
//                        arg.bin_area = 0;
//                        arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
//                    } else {
//                        for (int c = 0; c < c_block; c++) {
//                            dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] = 0;
//                        }
//                    }
//                } else {
//                    int top_y_index    = static_cast<int>(floorf(in_y));
//                    int bottom_y_index = static_cast<int>(ceilf(in_y));
//                    int left_x_index   = static_cast<int>(floorf(in_x));
//                    int right_x_index  = static_cast<int>(ceilf(in_x));
//
//                    if (right_x_index > jpp.iw - 1)
//                        right_x_index = jpp.iw - 1;
//
//                    if (bottom_y_index > jpp.ih - 1)
//                        bottom_y_index = jpp.ih - 1;
//
//                    if (roi_pooling_kernel) {
//                        arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
//
//                        arg.xf = in_x - left_x_index;
//                        arg.yf = in_y - top_y_index;
//
//                        arg.xoff = (size_t) ((right_x_index - left_x_index) * jpp.c_block * sizeof(float));
//                        arg.yoff = (size_t) ((bottom_y_index - top_y_index) * jpp.iw * jpp.c_block * sizeof(float));
//
//                        arg.src = &src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
//                                            top_y_index * src_strides[2] + left_x_index * src_strides[3]];
//                        arg.bin_area = 1;
//                    } else {
//                        for (int c = 0; c < 1; c++) {
//                            const float top_left     = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
//                                                                top_y_index * src_strides[2] + left_x_index * src_strides[3] + c];
//                            const float top_right    = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
//                                                                top_y_index * src_strides[2] + right_x_index * src_strides[3] + c];
//                            const float bottom_left  = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
//                                                                bottom_y_index * src_strides[2] + left_x_index * src_strides[3] + c];
//                            const float bottom_right = src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
//                                                                bottom_y_index * src_strides[2] + right_x_index * src_strides[3] + c];
//
//                            const float top    = top_left + (top_right - top_left) * (in_x - left_x_index);
//                            const float bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);
//
//                            dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] =
//                                    top + (bottom - top) * (in_y - top_y_index);
//                        }
//                    }
//                }
//            }
//
//            if (roi_pooling_kernel) {
//                (*roi_pooling_kernel)(&arg);
//            }
//        }
//    });

}


bool MKLDNNROIAlignNode::created() const {
    return getType() == ROIAlign;
}

void MKLDNNROIAlignNode::createPrimitive() {}

REG_MKLDNN_PRIM_FOR(MKLDNNROIAlignNode, ROIAlign);
