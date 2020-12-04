// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_roi_align_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_extension_utils.h>
#include <mkldnn_types.h>
#include <utils/bfloat16.hpp>
#include <cpu_isa_traits.hpp>
#include "ie_parallel.hpp"
#include <mkldnn_selective_build.h>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl::cpu;

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

    pooledH = genericLayer->GetParamAsInt("pooled_h");
    pooledW = genericLayer->GetParamAsInt("pooled_w");
    spatialScale = genericLayer->GetParamAsFloat("spatial_scale");
    samplingRatio = genericLayer->GetParamAsInt("sampling_ratio");
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

    Precision inputPrec = getCnnLayer()->insData[0].lock()->getPrecision();
    Precision outputPrec = getCnnLayer()->outData[0]->getPrecision();

    if (!mayiuse(avx512_core_bf16)) {
        if (outputPrec == Precision::BF16 || inputPrec == Precision::BF16)
            outputPrec = inputPrec = Precision::FP32;
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrec);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrec);

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

    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nchw);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nchw);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nchw});
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nhwc);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nhwc);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nhwc});
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw16c);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw16c);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw16c});
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw8c);
    config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::f32, memory::nc);
    config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::u8, memory::x);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nChw8c);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw8c});
}

namespace {
struct ROIAlignContext {
    MKLDNNROIAlignNode &node;
    mkldnn::stream strm;
};
}

template<typename T>
struct MKLDNNROIAlignNode::ROIAlignExecute {
    using srcT = typename std::tuple_element<0, T>::type;
    using dstT = typename std::tuple_element<1, T>::type;

    void operator()(ROIAlignContext & ctx) {
        ctx.node.executeSpecified<srcT, dstT>(ctx.strm);
    }
};
void MKLDNNROIAlignNode::execute(mkldnn::stream strm) {
    auto inputPrec = getParentEdgeAt(0)->getMemory().GetDescriptor().data.data_type;
    auto outputPrec = getChildEdgeAt(0)->getMemory().GetDescriptor().data.data_type;
    if (!((inputPrec == mkldnn_bf16 && outputPrec == mkldnn_bf16) ||
          (inputPrec == mkldnn_f32 && outputPrec == mkldnn_f32)))
        THROW_IE_EXCEPTION <<"ROIAlign doesn't support demanded precisions";

    ROIAlignContext ctx = {
            *this,
            strm
    };

    OV_SWITCH(MKLDNNPlugin, ROIAlignExecute, ctx, std::tie(inputPrec, outputPrec),
              OV_CASE2(mkldnn_f32, mkldnn_f32, float, float),
              OV_CASE2(mkldnn_bf16, mkldnn_bf16, bfloat16_t, bfloat16_t))
}

template <typename inputType, typename outputType>
void MKLDNNROIAlignNode::executeSpecified(mkldnn::stream strm) {
    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &srcMemory2 = getParentEdgeAt(2)->getMemory();

    auto &dstMemory = getChildEdgeAt(0)->getMemory();
    int blockSize;
    auto selectedFmt = srcMemory0.GetDescriptor().data.format;
    switch (selectedFmt) {
        case mkldnn_nchw:
        case mkldnn_nhwc:
            blockSize = 1;
            break;
        case mkldnn_nChw16c:
            blockSize = 16;
            break;
        case mkldnn_nChw8c:
            blockSize = 8;
            break;
        default:
            THROW_IE_EXCEPTION << "unsupported format for ROIAlign";
    }

    const auto *srcData = reinterpret_cast<const inputType *>(srcMemory0.GetData());
    const auto *srcRoi = reinterpret_cast<const float *>(srcMemory1.GetData());
    const auto *srcRoiIdx = reinterpret_cast<const int *>(srcMemory2.GetData());
    auto *dst = reinterpret_cast<outputType *>(dstMemory.GetData());

    auto nominalRoiCount = static_cast<int>(srcMemory1.GetDims()[0]);
    int realRois = 0;
    int C = static_cast<int>(srcMemory0.GetDims()[1]);
    int H = static_cast<int>(srcMemory0.GetDims()[2]);
    int W = static_cast<int>(srcMemory0.GetDims()[3]);

    const int binCount = pooledH * pooledH;

    int hInputCoeff;
    int wInputCoeff;
    int hOutputCoeff;
    int wOutputCoeff;
    if (selectedFmt == mkldnn_nhwc) {
        hInputCoeff = W * C;
        wInputCoeff = C;
        hOutputCoeff = pooledH * C;
        wOutputCoeff = C;
    } else {  // nchw, nChw16c, nChw8c
        hInputCoeff = W * blockSize;
        wInputCoeff = blockSize;
        hOutputCoeff = pooledH * blockSize;
        wOutputCoeff = blockSize;
    }

    for (; realRois < nominalRoiCount; realRois++) {
        const int *srcRoiIdxPtr = &srcRoiIdx[realRois];
        auto roiBatchInd = srcRoiIdxPtr[0];
        if (roiBatchInd == -1) {
            break;
        }
    }

    int roiOff;

    for (int n = 0; n < realRois; ++n) {
        roiOff = n * 4;
        const float* srcRoiPtr = &srcRoi[roiOff];
        const int* srcRoiIdxPtr = &srcRoiIdx[n];

        int roiBatchInd = srcRoiIdxPtr[0];
        if (roiBatchInd < -1) {
            THROW_IE_EXCEPTION << "Batch index cannot be less, than -1";
        } else if (roiBatchInd >= srcMemory0.GetDims()[0]) {
            THROW_IE_EXCEPTION << "Demanded batch (id = " << roiBatchInd << ") doesn't exist";
        }

        float x1 = srcRoiPtr[0] * spatialScale;
        float y1 = srcRoiPtr[1] * spatialScale;
        float x2 = srcRoiPtr[2] * spatialScale;
        float y2 = srcRoiPtr[3] * spatialScale;

        float roiHeight = std::max(y2 - y1, 1.0f);
        float roiWidth = std::max(x2 - x1, 1.0f);
        float binHeight = roiHeight / pooledH;
        float binWidth = roiWidth / pooledW;

        auto samplingRatioX = samplingRatio == 0 ? static_cast<int>(ceil(binWidth)) : samplingRatio;
        auto samplingRatioY = samplingRatio == 0 ? static_cast<int>(ceil(binHeight)) : samplingRatio;

        uint64_t numSamplesInBin = samplingRatioX * samplingRatioY;

        float sampleDistanceX = binWidth / samplingRatioX;
        float sampleDistanceY = binHeight / samplingRatioY;
        // prepare arrays for sampling points and weights
        std::vector<std::pair<int, int>> pointVector;
        std::vector<float> weightVector;
        pointVector.reserve(4 * numSamplesInBin * binCount);
        weightVector.reserve(4 * numSamplesInBin * binCount);

        for (int yBinInd = 0; yBinInd < pooledH; ++yBinInd) {
            for (int xBinInd = 0; xBinInd < pooledW; ++xBinInd) {
                // run into bin
                for (unsigned int ySampleInd = 0; ySampleInd < samplingRatioY;
                     ySampleInd++) {
                    float sampleY = y1 + yBinInd * binHeight +
                                     sampleDistanceY * (0.5f + ySampleInd);
                    for (int64_t xSampleInd = 0; xSampleInd < samplingRatioX;
                         xSampleInd++) {
                        float sampleX = x1 + xBinInd * binWidth +
                                         sampleDistanceX * (0.5f + xSampleInd);

                        if (sampleX < -1.0 || sampleX > W ||
                            sampleY < -1.0 || sampleY > H) {
                            // For this sample we save 4x point (0,0) with weight 0
                            pointVector.insert(pointVector.end(), 4, {0, 0});
                            weightVector.insert(weightVector.end(), 4, float{0});
                            continue;
                        }
                        sampleX = std::max(sampleX, float{0});
                        sampleY = std::max(sampleY, float{0});

                        auto sampleYLow = static_cast<unsigned int>(sampleY);
                        auto sampleXLow = static_cast<unsigned int>(sampleX);
                        unsigned int sampleYHigh;
                        unsigned int sampleXHigh;
                        if (sampleYLow >= H - 1) {
                            sampleYHigh = sampleYLow = H - 1;
                            sampleY = static_cast<float>(sampleYLow);
                        } else {
                            sampleYHigh = sampleYLow + 1;
                        }
                        if (sampleXLow >= H - 1) {
                            sampleXHigh = sampleXLow = W - 1;
                            sampleX = static_cast<float>(sampleXLow);
                        } else {
                            sampleXHigh = sampleXLow + 1;
                        }
                        pointVector.push_back({sampleYLow, sampleXLow});
                        pointVector.push_back({sampleYLow, sampleXHigh});
                        pointVector.push_back({sampleYHigh, sampleXLow});
                        pointVector.push_back({sampleYHigh, sampleXHigh});

                        // weight calculation for bilinear interpolation
                        auto ly = sampleY - sampleYLow;
                        auto lx = sampleX - sampleXLow;
                        auto hy = 1.0f - ly;
                        auto hx = 1.0f - lx;

                        weightVector.push_back(hy * hx);
                        weightVector.push_back(hy * lx);
                        weightVector.push_back(ly * hx);
                        weightVector.push_back(ly * lx);
                    }
                }
            }
        }
        parallel_for(C, [&](int c) {
            size_t binOffsetInput;
            size_t binOffsetOutput;
            const int blockResidual = c % blockSize;
            const int blockIdx = (c / blockSize) * blockSize;
            if (selectedFmt == mkldnn_nhwc) {
                binOffsetInput = roiBatchInd * C * H * W + c;
                binOffsetOutput = n * C * binCount + c;

            } else {  // nchw, nChw16c, nChw8c
                binOffsetInput = (roiBatchInd * C + blockIdx) * H * W;
                binOffsetOutput = (n * C + blockIdx) * binCount;
            }

            unsigned int sampleIndex = 0;
            for (int yBinInd = 0; yBinInd < pooledH; ++yBinInd) {
                for (int xBinInd = 0; xBinInd < pooledW; ++xBinInd) {
                    float pooledValue = 0;
                    for (unsigned int binSampleInd = 0;
                         binSampleInd < numSamplesInBin;
                         binSampleInd++) {
                        size_t part1Index = binOffsetInput + pointVector[sampleIndex].first * hInputCoeff +
                                            pointVector[sampleIndex].second * wInputCoeff + blockResidual;
                        float part1 = srcData[part1Index];
                        size_t part2Index = binOffsetInput + pointVector[sampleIndex + 1].first * hInputCoeff +
                                             pointVector[sampleIndex + 1].second * wInputCoeff + blockResidual;
                        float part2 = srcData[part2Index];
                        size_t part3Index = binOffsetInput + pointVector[sampleIndex + 2].first * hInputCoeff +
                                             pointVector[sampleIndex + 2].second * wInputCoeff + blockResidual;
                        float part3 = srcData[part3Index];
                        size_t part4Index = binOffsetInput + pointVector[sampleIndex + 3].first * hInputCoeff +
                                             pointVector[sampleIndex + 3].second * wInputCoeff + blockResidual;
                        float part4 = srcData[part4Index];

                        switch (opType) {
                            case ROIAlignOpType::Max:
                            {
                                float sampleValue = std::max(
                                        {weightVector[sampleIndex] * part1,
                                         weightVector[sampleIndex + 1] * part2,
                                         weightVector[sampleIndex + 2] * part3,
                                         weightVector[sampleIndex + 3] * part4});
                                pooledValue = sampleValue > pooledValue ? sampleValue : pooledValue;
                                break;
                            }
                            case ROIAlignOpType::Avg:
                            default:
                            {
                                float sampleValue =
                                        weightVector[sampleIndex] * part1 +
                                        weightVector[sampleIndex + 1] * part2 +
                                        weightVector[sampleIndex + 2] * part3 +
                                        weightVector[sampleIndex + 3] * part4;
                                pooledValue += sampleValue / numSamplesInBin;
                            }
                        }
                        sampleIndex += 4;
                    }
                    size_t dstIndex = binOffsetOutput + yBinInd * hOutputCoeff +
                                       xBinInd * wOutputCoeff + blockResidual;
                    dst[dstIndex] = pooledValue;
                }
            }
        });
    }
}

bool MKLDNNROIAlignNode::created() const {
    return getType() == ROIAlign;
}

void MKLDNNROIAlignNode::createPrimitive() {}

REG_MKLDNN_PRIM_FOR(MKLDNNROIAlignNode, ROIAlign)
