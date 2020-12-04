// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

enum ROIAlignOpType {
    Max,
    Avg,
    Bilinear
};



class MKLDNNROIAlignNode : public MKLDNNNode {
public:
    MKLDNNROIAlignNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNROIAlignNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    int pooled_h = 7;
    int pooled_w = 7;
    int sampling_ratio = 2;
    float spatial_scale = 1.0f;
    ROIAlignOpType opType = Max;

};

}  // namespace MKLDNNPlugin

