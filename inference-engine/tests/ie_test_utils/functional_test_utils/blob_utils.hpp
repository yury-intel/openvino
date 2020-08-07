﻿// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <type_traits>

#include <gtest/gtest.h>
#include <ngraph_functions/pass/convert_prc.hpp>
#include <common_test_utils/test_common.hpp>
#include "blob_factory.hpp"
#include "blob_transform.hpp"
#include "precision_utils.h"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace FuncTestUtils {
namespace Bf16TestUtils {
static float reducePrecisionBitwise(const float in);
static short reducePrecisionBitwiseS(const float in);
}  // namespace Bf16TestUtils

enum CompareType{
    ABS,
    REL,
    ABS_AND_REL  //  if absolute and relative differences are too high, an exception is thrown
};
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Pointer to considered blob
 * @param ref Pointer to reference blob
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparison
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
static void inline compareRawBuffers(const dType *res, const dType *ref,
                                     size_t resSize, size_t refSize,
                                     CompareType compareType, float thr1 = 0.01, float thr2 = 0.01,
                                     bool printData = false) {
    if (printData) {
        std::cout << "Reference results: " << std::endl;
        for (size_t i = 0; i < refSize; i++) {
            std::cout << ref[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Test results: " << std::endl;
        for (size_t i = 0; i < resSize; i++) {
            std::cout << res[i] << " ";
        }
        std::cout << std::endl;
    }

    switch (compareType) {
        case CompareType::ABS:
            for (size_t i = 0; i < refSize; i++) {
                float absDiff = std::abs(res[i] - ref[i]);
                ASSERT_LT(absDiff, thr1) << "Relative comparison of values ref: " << ref[i] << " and res: "
                                         << res[i] << " , index in blobs: " << i << " failed!";
            }
            break;
        case CompareType::REL:
            for (size_t i = 0; i < refSize; i++) {
                float absDiff = std::abs(res[i] - ref[i]);
                float relDiff = absDiff / std::max(res[i], ref[i]);
                ASSERT_LT(relDiff, thr2) << "Relative comparison of values ref: " << ref[i] << " and res: "
                                         << res[i] << " , index in blobs: " << i << " failed!";
            }
            break;
        case CompareType::ABS_AND_REL:
            for (size_t i = 0; i < refSize; i++) {
                float absDiff = std::abs(res[i] - ref[i]);
                if (absDiff > thr1) {
                    float relDiff = absDiff / std::max(res[i], ref[i]);
                    ASSERT_LT(relDiff, thr2) << "Comparison of values ref: " << ref[i] << " and res: "
                                             << res[i] << " , index in blobs: " << i << " failed!";
                }
            }
            break;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Pointer to considered blob
 * @param ref Pointer to reference blob
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData Flag if data printing is demanded
 */
template<typename dType>
static void inline compareRawBuffers(const dType *res, const dType *ref,
                                     size_t resSize, size_t refSize,
                                     float thr = 0.01,
                                     bool printData = false) {
    compareRawBuffers(res, ref, resSize, refSize, CompareType::ABS_AND_REL, thr, thr, printData);
}
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparision
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
static void inline compareRawBuffers(const std::vector<dType *> res, const std::vector<dType *> ref,
                                     const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                                     CompareType compareType,
                                     float thr1 = 0.01, float thr2 = 0.01, bool printData = false) {
    ASSERT_TRUE(res.size() == ref.size()) << "Reference and Results vector have to be same length";
    ASSERT_TRUE(res.size() == resSizes.size()) << "Results vector and elements count vector have to be same length";
    ASSERT_TRUE(ref.size() == refSizes.size()) << "Reference vector and elements count vector have to be same length";
    for (size_t i = 0; i < res.size(); i++) {
        if (printData) std::cout << "BEGIN CHECK BUFFER [" << i << "]" << std::endl;
        compareRawBuffers(res[i], ref[i], resSizes[i], refSizes[i], compareType, thr1, thr2, printData);
        if (printData) std::cout << "END CHECK BUFFER [" << i << "]" << std::endl;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
static void inline compareRawBuffers(const std::vector<dType *> res, const std::vector<dType *> ref,
                                     const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                                     float thr = 0.01, bool printData = false) {
    compareRawBuffers(res, ref, resSizes, refSizes, CompareType::ABS_AND_REL, thr, thr, printData);
}
/**
 * @brief Checks values of two blobs according to given algorithm and thresholds.
 * In ABS and REL cases thr1 corresponds to the single threshold,
 * In ABS_AND_REL case thr1 and thr2 mean absolute and relative threshold
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param compareType Defines an algorithm of comparision
 * @param thr1 First threshold of difference
 * @param thr2 Second threshold of difference
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
static void inline compareRawBuffers(const std::vector<dType *> res, const std::vector<std::shared_ptr<dType *>> ref,
                                     const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                                     CompareType compareType,
                                     float thr1 = 0.01, float thr2 = 0.01, bool printData = false) {
    ASSERT_TRUE(res.size() == ref.size()) << "Reference and Results vector have to be same length";
    ASSERT_TRUE(res.size() == resSizes.size()) << "Results vector and elements count vector have to be same length";
    ASSERT_TRUE(ref.size() == refSizes.size()) << "Reference vector and elements count vector have to be same length";
    for (size_t i = 0; i < res.size(); i++) {
        if (printData) std::cout << "BEGIN CHECK BUFFER [" << i << "]" << std::endl;
        compareRawBuffers(res[i], *ref[i], resSizes[i], refSizes[i], compareType, thr1, thr2, printData);
        if (printData) std::cout << "END CHECK BUFFER [" << i << "]" << std::endl;
    }
}
/**
 * @brief Checks absolute and relative difference of blob values according to given threshold.
 *
 * @tparam dType Type of blob data
 * @param res Vector of considered blob values
 * @param ref Vector of reference blob values
 * @param resSize Size of considered blob
 * @param refSize Size of reference blob
 * @param thr Threshold of difference, absolute and relative simultaneously
 * @param printData A flag if data printing is demanded
 */
template<typename dType>
static void inline compareRawBuffers(const std::vector<dType *> res, const std::vector<std::shared_ptr<dType *>> ref,
                                     const std::vector<size_t> &resSizes, const std::vector<size_t> &refSizes,
                                     float thr = 0.01, bool printData = false) {
    compareRawBuffers(res, ref, resSizes, refSizes, CompareType::ABS_AND_REL, thr, thr, printData);
}

template<InferenceEngine::Precision::ePrecision PRC>
void inline
compareBlobData(const InferenceEngine::Blob::Ptr &res, const InferenceEngine::Blob::Ptr &ref, float max_diff = 0.01,
                const std::string &assertDetails = "", bool printData = false) {
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    const dataType *res_ptr = res->cbuffer().as<dataType *>();
    size_t res_size = res->byteSize();

    const dataType *ref_ptr = ref->cbuffer().as<dataType *>();
    size_t ref_size = ref->byteSize();

    ASSERT_EQ(res_size, ref_size) << "Comparing blobs have different size. " << assertDetails;
    if (printData) {
        std::cout << "Reference results: " << std::endl;
        for (size_t i = 0; i < ref_size / sizeof(dataType); i++) {
            std::cout << ref_ptr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Test results: " << std::endl;
        for (size_t i = 0; i < res_size / sizeof(dataType); i++) {
            std::cout << res_ptr[i] << " ";
        }
        std::cout << std::endl;
    }

    for (size_t i = 0; i < ref_size / sizeof(dataType); i++) {
        auto resVal = PRC == InferenceEngine::Precision::FP16 ? InferenceEngine::PrecisionUtils::f16tof32(res_ptr[i])
                                                              : res_ptr[i];
        auto refVal = PRC == InferenceEngine::Precision::FP16 ? InferenceEngine::PrecisionUtils::f16tof32(ref_ptr[i])
                                                              : ref_ptr[i];
        float absDiff = std::abs(resVal - refVal);
        if (absDiff > max_diff) {
            float relDiff = absDiff / std::max(res_ptr[i], ref_ptr[i]);
            ASSERT_LT(relDiff, max_diff) << "Relative comparison of values ref: " << ref_ptr[i] << " and res: "
                                         << res_ptr[i] << " , index in blobs: " << i << " failed!" << assertDetails;
        }
    }
}


template<InferenceEngine::Precision::ePrecision PRC>
void inline
compareBlobData(const std::vector<InferenceEngine::Blob::Ptr> &res, const std::vector<InferenceEngine::Blob::Ptr> &ref,
                float max_diff = 0.01,
                const std::string &assertDetails = "", bool printData = false) {
    IE_ASSERT(res.size() == ref.size()) << "Length of comparing and references blobs vector are not equal!"
                                        << assertDetails;
    using dataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    for (size_t i = 0; i < res.size(); i++) {
        if (printData)
            std::cout << "BEGIN CHECK BLOB [" << i << "]" << std::endl;
        compareBlobData<PRC>(res[i], ref[i], max_diff, assertDetails, printData);
        if (printData)
            std::cout << "END CHECK BLOB [" << i << "]" << std::endl;
    }
}

void inline
compareBlobs(const InferenceEngine::Blob::Ptr &res, const InferenceEngine::Blob::Ptr &ref, float max_diff = 0.01,
             const std::string &assertDetails = "", bool printData = false) {
    ASSERT_EQ(res->byteSize(), ref->byteSize()) << "Blobs have different byteSize(): "
                                                << res->byteSize() << " and " << ref->byteSize();

    ASSERT_EQ(res->getTensorDesc(), ref->getTensorDesc()) << "Blobs have different TensorDesc()";

    switch (res->getTensorDesc().getPrecision()) {
#define COMPARE_WITH_REF(TYPE) case TYPE: { \
                                      FuncTestUtils::compareBlobData<TYPE>(res, \
                                                                             ref, \
                                                                             max_diff, \
                                                                             assertDetails, \
                                                                             printData); break; }
        COMPARE_WITH_REF(InferenceEngine::Precision::FP32);
        COMPARE_WITH_REF(InferenceEngine::Precision::FP16);
        COMPARE_WITH_REF(InferenceEngine::Precision::I64);
#undef COMPARE_WITH_REF
        default:
            THROW_IE_EXCEPTION << "Precision " << res->getTensorDesc().getPrecision().name()
                               << " is not covered by FuncTestUtils::compareBlobs() method";
    }
}

void inline GetComparisonThreshold(InferenceEngine::Precision prc, float &absoluteThreshold, float &relativeThreshold) {
    switch (prc) {
        case InferenceEngine::Precision::FP32:
            absoluteThreshold = relativeThreshold = 1e-4;
            break;
        case InferenceEngine::Precision::FP16:
            absoluteThreshold = relativeThreshold = 1e-2;
            break;
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::I8:
        case InferenceEngine::Precision::U8:
            absoluteThreshold = relativeThreshold = 1;
            break;
        default:
            THROW_IE_EXCEPTION << "Unhandled precision " << prc << " passed to the GetComparisonThreshold()";
    }
}

float inline GetComparisonThreshold(InferenceEngine::Precision prc) {
    float res;
    GetComparisonThreshold(prc, res, res);
    return res;
}

// Copy from net_pass.h
template<InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
void inline convertArrayPrecision(typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type *dst,
                                  const typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type *src,
                                  size_t nelem) {
    using dst_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = static_cast<dst_type>(src[i]);
    }
}

template<>
void inline
convertArrayPrecision<InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32>(float *dst, const short *src,
                                                                                          size_t nelem) {
    uint16_t a = *reinterpret_cast<const uint16_t *>(src);
    InferenceEngine::PrecisionUtils::f16tof32Arrays(dst, src, nelem, 1.0f, 0.0f);
}

template<InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
InferenceEngine::Blob::Ptr inline convertBlobPrecision(const InferenceEngine::Blob::Ptr &blob) {
    using from_d_type = typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type;
    using to_d_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    auto tensor_desc = blob->getTensorDesc();
    InferenceEngine::Blob::Ptr new_blob = InferenceEngine::make_shared_blob<to_d_type>(
            InferenceEngine::TensorDesc{PREC_TO, tensor_desc.getDims(), tensor_desc.getLayout()});
    new_blob->allocate();
    auto target = new_blob->buffer().as<to_d_type *>();
    auto source = blob->buffer().as<from_d_type *>();
    convertArrayPrecision<PREC_FROM, PREC_TO>(target, source, blob->size());
    return new_blob;
}
// Copy from net_pass.h


template<InferenceEngine::Precision::ePrecision targetPRC>
InferenceEngine::Blob::Ptr inline copyBlobWithCast(const InferenceEngine::Blob::Ptr &blob) {
    InferenceEngine::Blob::Ptr newBlob;
    switch (blob->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::FP32, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::FP16:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::FP16, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I16:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::I16, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I8:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::I8, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::U8:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::U8, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I32:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::I32, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::BOOL:
            newBlob = FuncTestUtils::convertBlobPrecision<InferenceEngine::Precision::BOOL, targetPRC>(blob);
            break;
        default:
            THROW_IE_EXCEPTION << "Conversion from blob with precision " << blob->getTensorDesc().getPrecision().name()
                               << " not implemented yet!";
    }
    return newBlob;
}

InferenceEngine::Blob::Ptr inline createAndFillBlobFloatNormalDistribution(const InferenceEngine::TensorDesc &td,
                                                                           const float mean,
                                                                           const float stddev,
                                                                           const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_normal_random_float<X>(blob, mean, stddev, seed); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            THROW_IE_EXCEPTION << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

InferenceEngine::Blob::Ptr inline createAndFillBlobFloat(const InferenceEngine::TensorDesc &td,
                                                         const uint32_t range = 10,
                                                         const int32_t start_from = 0,
                                                         const int32_t resolution = 1,
                                                         const int32_t seed = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);

    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_random_float<X>(blob, range, start_from, resolution, seed); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            THROW_IE_EXCEPTION << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

InferenceEngine::Blob::Ptr inline createAndFillBlobWithFloatArray(const InferenceEngine::TensorDesc &td,
                                                                  const float values[],
                                                                  const int size) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_float_array<X>(blob, values, size); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            THROW_IE_EXCEPTION << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

InferenceEngine::Blob::Ptr inline createAndFillBlob(const InferenceEngine::TensorDesc &td,
                                                    const uint32_t range = 10,
                                                    const int32_t start_from = 0,
                                                    const int32_t resolution = 1) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_random<X>(blob, range, start_from, resolution); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            THROW_IE_EXCEPTION << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

InferenceEngine::Blob::Ptr inline createAndFillBlobConsistently(
        const InferenceEngine::TensorDesc &td,
        const uint32_t range,
        const int32_t start_from,
        const int32_t resolution) {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(td);
    blob->allocate();
    switch (td.getPrecision()) {
#define CASE(X) case X: CommonTestUtils::fill_data_consistently<X>(blob, range, start_from, resolution); break;
        CASE(InferenceEngine::Precision::FP32)
        CASE(InferenceEngine::Precision::FP16)
        CASE(InferenceEngine::Precision::U8)
        CASE(InferenceEngine::Precision::U16)
        CASE(InferenceEngine::Precision::I8)
        CASE(InferenceEngine::Precision::I16)
        CASE(InferenceEngine::Precision::I64)
        CASE(InferenceEngine::Precision::BIN)
        CASE(InferenceEngine::Precision::I32)
        CASE(InferenceEngine::Precision::BOOL)
#undef CASE
        default:
            THROW_IE_EXCEPTION << "Wrong precision specified: " << td.getPrecision().name();
    }
    return blob;
}

InferenceEngine::Blob::Ptr inline convertBlobLayout(const InferenceEngine::Blob::Ptr& in,
                                                    InferenceEngine::Layout layout) {
    IE_ASSERT(in != nullptr) << "Got NULL pointer";

    const auto& inDesc = in->getTensorDesc();

    if (inDesc.getLayout() == layout) {
        return in;
    }

    const auto outDesc = InferenceEngine::TensorDesc(inDesc.getPrecision(), inDesc.getDims(), layout);

    const auto out = make_blob_with_precision(outDesc);
    out->allocate();

    InferenceEngine::blob_copy(in, out);

    return out;
}

template<typename dType>
static void fillInputsBySinValues(dType* data, size_t size) {
    if (std::is_same<dType, float>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sin(static_cast<float>(i));
        }
    } else if (std::is_same<dType, short>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(sin(static_cast<float>(i)));
        }
    }
}

template<typename dType>
static void fillInputsByCosValues(dType* data, size_t size) {
    if (std::is_same<dType, float>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sin(static_cast<float>(i));
        }
    } else if (std::is_same<dType, short>::value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(sin(static_cast<float>(i)));
        }
    }
}

static int fillInputsBySinValues(InferenceEngine::Blob::Ptr blob) {
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        return -1;
    }
    if (mblob->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
        return -2;
    }
    auto lm = mblob->rwmap();
    fillInputsBySinValues(lm.as<float*>(), mblob->size());
    return 0;
}

static int fillInputsByCosValues(InferenceEngine::Blob::Ptr blob) {
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        return -1;
    }
    if (mblob->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
        return -2;
    }
    auto lm = mblob->rwmap();
    fillInputsByCosValues(lm.as<float*>(), mblob->size());
    return 0;
}
enum RefMode {
    INTERPRETER,
    CONSTANT_FOLDING,
    IE
};

class IComparableNGTestCommon : public CommonTestUtils::TestsCommon {
protected:
    virtual void Prepare() {}
    virtual void SetInput() {}
    virtual void Preproc() {}
    virtual void Infer() = 0;
    virtual void Postproc() {}
    virtual void Validate() {}
    void Run() final {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        Prepare();
        SetInput();
        Preproc();
        Infer();
        Postproc();
        Validate();
    }
};
class ComparableNGTestCommon : public IComparableNGTestCommon {
protected:
    float threshold;
    FuncTestUtils::RefMode refMode = FuncTestUtils::RefMode::INTERPRETER;

    std::shared_ptr<ngraph::Function> function;

    std::vector<std::vector<std::uint8_t>> referenceInputs;
    std::vector<std::vector<std::uint8_t>> expectedOutputs;

    void SetRefMode(FuncTestUtils::RefMode mode) {
        refMode = mode;
    }

    FuncTestUtils::RefMode GetRefMode() {
        return refMode;
    }

    template<class T>
    void CompareTypedVectors(const std::vector<std::vector<T>>& v1, const std::vector<std::vector<T>>& v2) {
        InferenceEngine::Precision precision;
        ASSERT_EQ(v1.size(), v2.size());
        auto size = v1.size();
        for (std::size_t idx = 0; idx < size; ++idx) {
            const auto& expected = v1[idx];
            const auto& actual = v2[idx];
            if (std::is_same<T, int>::value) {
                precision = InferenceEngine::Precision::I32;
            } else if (std::is_same<T, float>::value) {
                precision = InferenceEngine::Precision::FP32;
            } else {
                THROW_IE_EXCEPTION << "Precision not supported";
            }

            CompareValues(v1[idx].data(), v2[idx].data(), v1.size(), precision);
        }
    }
    template <class T>
    struct FillingBoundaries {
        T left;
        T right;
        FillingBoundaries() {
            if (std::is_same<T, int>::value) {
                left = defaultLeftIntBoundary;
                right = defaultRightIntBoundary;
            } else if (std::is_same<T, float>::value) {
                left = defaultLeftFloatBoundary;
                right = defaultRightFloatBoundary;
            }
        }
        FillingBoundaries(T _l, T _r): left(_l), right(_r) {}
    private:
        int defaultLeftIntBoundary = 0;
        int defaultRightIntBoundary = 127;
        float defaultLeftFloatBoundary = -10.0f;
        float defaultRightFloatBoundary = 10.0f;
    };
    template<class T>
    void FillByteVectorAsTyped(std::vector<std::vector<uint8_t>>& byteVector, FillingBoundaries<T> fb = FillingBoundaries<T>()) {
        for (auto vecIt = byteVector.begin(); vecIt != byteVector.end(); ++vecIt) {
            int modulo4 = 0;
            T range = fb.right - fb.left;
            auto size = vecIt->size();
            float shift;
            T typedShift;
            T currTValue = fb.left;
            uint8_t currByteValue;
            unsigned char currUChar;
            for (auto byteIt = vecIt->begin(); byteIt != vecIt->end(); byteIt++) {
                switch (modulo4 % 4) {
                    case 0: currUChar = (unsigned char)currTValue;
                        break;
                    case 1: currUChar = (unsigned char)currTValue >> 8;
                        break;
                    case 2: currUChar = (unsigned char)currTValue >> 16;
                        break;
                    case 3:
                        shift = (static_cast<float>(range) / ((size / 4) - 1)) * ((modulo4 / 4) + 1);
                        typedShift = static_cast<T>(shift);
                        currTValue = fb.left + typedShift;
                        currUChar = (unsigned char)currTValue >> 24;
                        break;
                }
                currByteValue = reinterpret_cast<uint8_t>(currUChar);
                *byteIt = currByteValue;
                modulo4++;
            }
        }
    }
    void CompareBytes(const std::vector<std::vector<std::uint8_t>>& expectedVector, const std::vector<std::vector<std::uint8_t>>& actualVector,
                         const InferenceEngine::Precision precision) {
        for (std::size_t idx = 0; idx < expectedVector.size(); ++idx) {
            const auto& expected = expectedVector[idx];
            const auto& actual = actualVector[idx];
            ASSERT_EQ(expectedVector.size(), actualVector.size());
            const unsigned char *expectedBuffer = expected.data();
            const unsigned char *actualBuffer = actual.data();
            auto size = actual.size();
            CompareValues(expectedBuffer, actualBuffer, size, precision);
        }
    }
    template<class T>
    void CompareValues(const T *expected, const T *actual, std::size_t size, T thr) {
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

    void CompareValues(const void *expected, const void *actual, std::size_t size, const InferenceEngine::Precision precision) {
        switch (precision) {
            case InferenceEngine::Precision::FP32:
                FuncTestUtils::ComparableNGTestCommon::CompareValues(
                        reinterpret_cast<const float *>(expected), reinterpret_cast<const float *>(actual),
                        size, threshold);
                break;
            case InferenceEngine::Precision::I32:
                FuncTestUtils::ComparableNGTestCommon::CompareValues(
                        reinterpret_cast<const std::int32_t *>(expected),
                        reinterpret_cast<const std::int32_t *>(actual), size, 0);
                break;
            default:
                FAIL() << "Comparator for " << precision << " precision isn't supported";
        }
    }

    virtual std::vector<std::vector<std::uint8_t>> CalculateRefs(std::shared_ptr<ngraph::Function> _function, std::vector<std::vector<std::uint8_t>> _inputs) {
        // nGraph interpreter does not support f16
        // IE converts f16 to f32
        ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(_function);
        _function->validate_nodes_and_infer_types();
        return ngraph::helpers::interpreterFunction(_function, _inputs, ::ngraph::element::Type_t::undefined);
    }

    static void FillByteVectorRandomly(std::vector<std::vector<std::uint8_t>> &inputVector, unsigned seed = 1) {
        int modulo4 = 0;
        int randInt;
        unsigned char randUChar;
        uint8_t randByte;
        for (auto vecIt = inputVector.begin(); vecIt != inputVector.end(); vecIt++) {
            for (auto byteIt = vecIt->begin(); byteIt != vecIt->end(); byteIt++) {
                switch (modulo4 % 4) {
                    case 0: randInt = rand_r(&seed);
                        randUChar = (unsigned char)randInt;
                        break;
                    case 1: randUChar = (unsigned char)randInt >> 8;
                        break;
                    case 2: randUChar = (unsigned char)randInt >> 16;
                        break;
                    case 3: randUChar = (unsigned char)randInt >> 24;
                        break;
                }
                randByte = reinterpret_cast<uint8_t>(randUChar);
                *byteIt = randByte;
                modulo4++;
            }
        }
    }

    template<class T>
    static std::vector<std::vector<T>> getTypedVector(std::vector<std::vector<std::uint8_t>> &byteVector) {
        size_t ratio;
        if (std::is_same<T, int>::value || std::is_same<T, float>::value) {
            ratio = 4;
        } else {
            THROW_IE_EXCEPTION << "Unsupported precision";
        }
        std::vector<std::vector<T>> typedVector;

        for (auto &vec : byteVector) {
            if (vec.size() % ratio != 0) {
                THROW_IE_EXCEPTION << "Byte vector doesn't match given vector type";
            }
            std::vector<T> innerVector(vec.size() / ratio);
            memcpy(static_cast<void *>(innerVector.data()), static_cast<void *>(vec.data()),
                    vec.size());
            typedVector.push_back(innerVector);
        }
        return typedVector;
    }
};

namespace Bf16TestUtils {
static float reducePrecisionBitwise(const float in) {
    float f = in;
    int* i = reinterpret_cast<int*>(&f);
    int t2 = *i & 0xFFFF0000;
    float ft1 = *(reinterpret_cast<float*>(&t2));
    if ((*i & 0x8000) && (*i & 0x007F0000) != 0x007F0000) {
        t2 += 0x10000;
        ft1 = *(reinterpret_cast<float*>(&t2));
    }
    return ft1;
}

static short reducePrecisionBitwiseS(const float in) {
    float f = reducePrecisionBitwise(in);
    int intf = *reinterpret_cast<int*>(&f);
    intf = intf >> 16;
    short s = intf;
    return s;
}
}  // namespace Bf16TestUtils
}  // namespace FuncTestUtils
