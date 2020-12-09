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
#include <mkldnn_types.h>
#include "ie_parallel.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;

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

    if (getParentEdges().size() != 3)
        THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 0th input with rank: " << getParentEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims().ndims() != 2) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getParentEdgeAt(2)->getDims().ndims() != 1) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getChildEdgeAt(0)->getDims().ndims() != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "doesn't support output with rank: " << getChildEdgeAt(0)->getDims().ndims();
    }

    if (getParentEdgeAt(1)->getDims()[1] != 4) {
        THROW_IE_EXCEPTION << errorPrefix << "has invalid shape on 1st input: ["
                           << getParentEdgeAt(1)->getDims()[0] << "," << getParentEdgeAt(1)->getDims()[1] << "]";
    }

    pooled_h = genericLayer->GetParamAsInt("pooled_h");
    pooled_w = genericLayer->GetParamAsInt("pooled_w");
    spatial_scale = genericLayer->GetParamAsFloat("spatial_scale");
    sampling_ratio = genericLayer->GetParamAsInt("sampling_ratio");
    std::string m = genericLayer->GetParamAsString("mode", "max");
    if (m == "max") {
        opType = ROIAlignOpType::Max;
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
    config.inConfs.resize(3);
    config.inConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.inConfs[1].constant = false;
    config.inConfs[1].inPlace = -1;
    config.inConfs[2].constant = false;
    config.inConfs[2].inPlace = -1;

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
    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::f32, memory::nchw);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nchw});
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::f32, memory::nChw16c);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::f32, memory::nChw16c);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw16c});
//    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::f32, memory::nChw8c);
//    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
//    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
//    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::f32, memory::nChw8c);
//    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw8c});
}

void MKLDNNROIAlignNode::execute(mkldnn::stream strm) {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &srcMemory2 = getParentEdgeAt(2)->getMemory();

    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    auto block_size = srcMemory0.GetDescriptor().data.format == mkldnn_nchw ? 1 :
            srcMemory0.GetDescriptor().data.format == mkldnn_nChw16c ? 16 : 8;

    const auto *src_data = reinterpret_cast<const float *>(srcMemory0.GetData());
    const auto *src_roi = reinterpret_cast<const float *>(srcMemory1.GetData());
    const auto *src_roi_idx = reinterpret_cast<const int *>(srcMemory2.GetData());

    float *dst = reinterpret_cast<float *>(dstMemory.GetData());

    auto config = getSelectedPrimitiveDescriptor()->getConfig();

    auto nominal_roi_count = static_cast<int>(srcMemory1.GetDims()[0]);
    int real_rois = 0;
    int C = static_cast<int>(srcMemory0.GetDims()[1]);
    int H = static_cast<int>(srcMemory0.GetDims()[2]);
    int W = static_cast<int>(srcMemory0.GetDims()[3]);

    const int block_offset_input = W * block_size;
    const int block_offset_output = pooled_w * block_size;
    const int bin_count = pooled_h * pooled_w;

    for (; real_rois < nominal_roi_count; real_rois++) {
        const int *src_roi_idx_ptr = &src_roi_idx[real_rois];
        auto roi_batch_ind = src_roi_idx_ptr[0];
        if (roi_batch_ind == -1) {
            break;
        }
    }

    int roi_off;

    for (int n = 0; n < real_rois; ++n) {
        roi_off = n * 4;
        const float* src_roi_ptr = &src_roi[roi_off];
        const int* src_roi_idx_ptr = &src_roi_idx[n];

        int roi_batch_ind = src_roi_idx_ptr[0];
        if (roi_batch_ind < -1) {
            THROW_IE_EXCEPTION << "Batch index cannot be less, than -1";
        } else if (roi_batch_ind >= srcMemory0.GetDims()[0]) {
            THROW_IE_EXCEPTION << "Demanded batch (id = "<< roi_batch_ind << ") doesn't exist";
        }

        float x1 = src_roi_ptr[0] * spatial_scale;
        float y1 = src_roi_ptr[1] * spatial_scale;
        float x2 = src_roi_ptr[2] * spatial_scale;
        float y2 = src_roi_ptr[3] * spatial_scale;

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
        point_vector.reserve(4 * num_samples_in_bin * bin_count);
        weight_vector.reserve(4 * num_samples_in_bin * bin_count);

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
                            sample_y < -1.0 || sample_y > H) {
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
                        if (sample_y_low >= H - 1) {
                            sample_y_high = sample_y_low = H - 1;
                            sample_y = static_cast<float>(sample_y_low);
                        } else {
                            sample_y_high = sample_y_low + 1;
                        }
                        if (sample_x_low >= H - 1) {
                            sample_x_high = sample_x_low = W - 1;
                            sample_x = static_cast<float>(sample_x_low);
                        } else {
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
        parallel_for(C, [&](int c) {
            const int block_residual = c % block_size;
            const int block_idx = (c / block_size) * block_size;
            size_t bin_offset_input = (roi_batch_ind * C + block_idx) * H * W;
            size_t bin_offset_output = (n * C + block_idx) * bin_count;
            unsigned int sample_index = 0;
            for (int y_bin_ind = 0; y_bin_ind < pooled_h; ++y_bin_ind) {
                for (int x_bin_ind = 0; x_bin_ind < pooled_w; ++x_bin_ind) {
                    float pooled_value = 0;
                    for (unsigned int bin_sample_ind = 0;
                         bin_sample_ind < num_samples_in_bin;
                         bin_sample_ind++) {
                        size_t part1_index = bin_offset_input + point_vector[sample_index].first * block_offset_input +
                                             point_vector[sample_index].second * block_size + block_residual;
                        float part1 = src_data[part1_index];
                        size_t part2_index = bin_offset_input + point_vector[sample_index + 1].first * block_offset_input +
                                             point_vector[sample_index + 1].second * block_size + block_residual;
                        float part2 = src_data[part2_index];
                        size_t part3_index = bin_offset_input + point_vector[sample_index + 2].first * block_offset_input +
                                             point_vector[sample_index + 2].second * block_size + block_residual;
                        float part3 = src_data[part3_index];
                        size_t part4_index = bin_offset_input + point_vector[sample_index + 3].first * block_offset_input +
                                             point_vector[sample_index + 3].second * block_size + block_residual;
                        float part4 = src_data[part4_index];

                        switch (opType) {
                            case ROIAlignOpType::Max:
                            {
                                float sample_value = std::max(
                                        {weight_vector[sample_index] * part1,
                                         weight_vector[sample_index + 1] * part2,
                                         weight_vector[sample_index + 2] * part3,
                                         weight_vector[sample_index + 3] * part4});
                                pooled_value = sample_value > pooled_value ? sample_value : pooled_value;
                                break;
                            }
                            case ROIAlignOpType::Avg:
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
                    size_t dst_index = bin_offset_output + y_bin_ind * block_offset_output +
                                       x_bin_ind * block_size + block_residual;
                    dst[dst_index] = pooled_value;
                }
            }
        });
    }
}


bool MKLDNNROIAlignNode::created() const {
    return getType() == ROIAlign;
}

void MKLDNNROIAlignNode::createPrimitive() {}

REG_MKLDNN_PRIM_FOR(MKLDNNROIAlignNode, ROIAlign);
