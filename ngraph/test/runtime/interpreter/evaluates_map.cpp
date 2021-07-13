// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

#include "backend.hpp"
#include "ngraph/ops.hpp"

#include <ngraph/runtime/reference/abs.hpp>
#include <ngraph/runtime/reference/adaptive_avg_pool.hpp>
#include <ngraph/runtime/reference/adaptive_max_pool.hpp>
#include <ngraph/runtime/reference/avg_pool.hpp>
#include <ngraph/runtime/reference/batch_norm.hpp>
#include <ngraph/runtime/reference/binary_convolution.hpp>
#include <ngraph/runtime/reference/bucketize.hpp>
#include <ngraph/runtime/reference/ceiling.hpp>
#include <ngraph/runtime/reference/convert.hpp>
#include <ngraph/runtime/reference/convolution.hpp>
#include <ngraph/runtime/reference/convolution_backprop_data.hpp>
#include <ngraph/runtime/reference/ctc_greedy_decoder.hpp>
#include <ngraph/runtime/reference/ctc_greedy_decoder_seq_len.hpp>
#include <ngraph/runtime/reference/ctc_loss.hpp>
#include <ngraph/runtime/reference/cum_sum.hpp>
#include <ngraph/runtime/reference/deformable_convolution.hpp>
#include <ngraph/runtime/reference/deformable_psroi_pooling.hpp>
#include <ngraph/runtime/reference/detection_output.hpp>
#include <ngraph/runtime/reference/einsum.hpp>
#include <ngraph/runtime/reference/elu.hpp>
#include <ngraph/runtime/reference/embedding_bag_offsets_sum.hpp>
#include <ngraph/runtime/reference/embedding_bag_packed_sum.hpp>
#include <ngraph/runtime/reference/embedding_segments_sum.hpp>
#include <ngraph/runtime/reference/experimental_detectron_detection_output.hpp>
#include <ngraph/runtime/reference/experimental_detectron_prior_grid_generator.hpp>
#include <ngraph/runtime/reference/experimental_detectron_topk_rois.hpp>
#include <ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp>
#include <ngraph/runtime/reference/extract_image_patches.hpp>
#include <ngraph/runtime/reference/fake_quantize.hpp>
#include <ngraph/runtime/reference/fft.hpp>
#include <ngraph/runtime/reference/gather_elements.hpp>
#include <ngraph/runtime/reference/gather_nd.hpp>
#include <ngraph/runtime/reference/gather_tree.hpp>
#include <ngraph/runtime/reference/gelu.hpp>
#include <ngraph/runtime/reference/grn.hpp>
#include <ngraph/runtime/reference/group_convolution.hpp>
#include <ngraph/runtime/reference/group_convolution_backprop_data.hpp>
#include <ngraph/runtime/reference/gru_cell.hpp>
#include <ngraph/runtime/reference/hard_sigmoid.hpp>
#include <ngraph/runtime/reference/log_softmax.hpp>
#include <ngraph/runtime/reference/lrn.hpp>
#include <ngraph/runtime/reference/lstm_cell.hpp>
#include <ngraph/runtime/reference/matrix_nms.hpp>
#include <ngraph/runtime/reference/mod.hpp>
#include <ngraph/runtime/reference/multiclass_nms.hpp>
#include <ngraph/runtime/reference/mvn.hpp>
#include <ngraph/runtime/reference/non_max_suppression.hpp>
#include <ngraph/runtime/reference/normalize_l2.hpp>
#include <ngraph/runtime/reference/pad.hpp>
#include <ngraph/runtime/reference/prior_box.hpp>
#include <ngraph/runtime/reference/proposal.hpp>
#include <ngraph/runtime/reference/psroi_pooling.hpp>
#include <ngraph/runtime/reference/region_yolo.hpp>
#include <ngraph/runtime/reference/reorg_yolo.hpp>
#include <ngraph/runtime/reference/reverse_sequence.hpp>
#include <ngraph/runtime/reference/rnn_cell.hpp>
#include <ngraph/runtime/reference/roi_pooling.hpp>
#include <ngraph/runtime/reference/roll.hpp>
#include <ngraph/runtime/reference/scatter_nd_update.hpp>
#include <ngraph/runtime/reference/select.hpp>
#include <ngraph/runtime/reference/selu.hpp>
#include <ngraph/runtime/reference/sequences.hpp>
#include <ngraph/runtime/reference/sign.hpp>
#include <ngraph/runtime/reference/squared_difference.hpp>
#include <ngraph/runtime/reference/tensor_iterator.hpp>
#include <ngraph/runtime/reference/utils/nms_common.hpp>

using namespace ngraph;
using namespace std;

namespace
{
    template <element::Type_t ET>
    bool evaluate(shared_ptr<Node> op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        return false;
    }

    namespace bucketize_v3
    {
        template <element::Type_t t1, element::Type_t t2, element::Type_t t3>
        inline void evaluate(const shared_ptr<op::v3::Bucketize>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            using T3 = typename element_type_traits<t3>::value_type;

            runtime::reference::bucketize<T1, T2, T3>(inputs[0]->get_data_ptr<T1>(),
                                                      inputs[1]->get_data_ptr<T2>(),
                                                      outputs[0]->get_data_ptr<T3>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_with_right_bound());
        }

        static inline constexpr uint16_t getElementMask(element::Type_t type1,
                                                        element::Type_t type2)
        {
            return (static_cast<uint8_t>(type1)) | (static_cast<uint8_t>(type2) << 8);
        }

    } // namespace bucketize_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::Bucketize>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (bucketize_v3::getElementMask(op->get_input_element_type(0),
                                             op->get_input_element_type(1)))
        {
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f32, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::f16, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::f16, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::f16, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::f16, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::f32, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i32, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::i32, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::f32):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::f32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::f16):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::f16, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::i32):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::i32, ET>(
                op, outputs, inputs);
            break;
        case bucketize_v3::getElementMask(element::Type_t::i64, element::Type_t::i64):
            bucketize_v3::evaluate<element::Type_t::i64, element::Type_t::i64, ET>(
                op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Convolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        runtime::reference::convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end());
        return true;
    }

    namespace bin_conv_v1
    {
        template <element::Type_t t_in, element::Type_t t_f>
        inline void evaluate(const shared_ptr<op::v1::BinaryConvolution>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T_IN = typename element_type_traits<t_in>::value_type;
            using T_F = typename element_type_traits<t_f>::value_type;

            const auto in_data_ptr = inputs[0]->get_data_ptr<T_IN>();
            const auto filter_data_ptr = inputs[1]->get_data_ptr<T_F>();
            auto out_data_ptr = outputs[0]->get_data_ptr<T_IN>();
            const auto in_shape = inputs[0]->get_shape();
            const auto filter_shape = inputs[1]->get_shape();
            const auto out_shape = outputs[0]->get_shape();

            runtime::reference::binary_convolution<T_IN, T_F>(in_data_ptr,
                                                              filter_data_ptr,
                                                              out_data_ptr,
                                                              in_shape,
                                                              filter_shape,
                                                              out_shape,
                                                              op->get_strides(),
                                                              op->get_dilations(),
                                                              op->get_pads_begin(),
                                                              op->get_pads_end(),
                                                              op->get_pad_value());
        }
    } // bin_conv_v1

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::BinaryConvolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::u1:
            bin_conv_v1::evaluate<ET, element::Type_t::u8>(op, outputs, inputs);
            break;
        default:
            throw std::runtime_error(
                "BinaryConvolution supports only u1 element type for filters input");
            break;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::ConvolutionBackpropData>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution_backprop_in<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            in_dilation,
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_strides(),
            op->get_output_padding());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        runtime::reference::group_convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolutionBackpropData>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto filter_data_ptr = inputs[1]->get_data_ptr<ET>();
        const auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_shape = inputs[0]->get_shape();
        const auto filter_shape = inputs[1]->get_shape();
        const auto out_shape = outputs[0]->get_shape();
        runtime::reference::group_convolution_backprop_data<
            typename element_type_traits<ET>::value_type>(in_data_ptr,
                                                          filter_data_ptr,
                                                          out_data_ptr,
                                                          in_shape,
                                                          filter_shape,
                                                          out_shape,
                                                          op->get_strides(),
                                                          op->get_dilations(),
                                                          op->get_pads_begin(),
                                                          op->get_pads_end(),
                                                          op->get_output_padding());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::DeformableConvolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto offset_data_ptr = inputs[1]->get_data_ptr<ET>();
        const auto filter_data_ptr = inputs[2]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& offset_shape = inputs[1]->get_shape();
        const auto& filter_shape = inputs[2]->get_shape();
        runtime::reference::deformable_convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            offset_data_ptr,
            filter_data_ptr,
            out_data_ptr,
            in_shape,
            offset_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_group(),
            op->get_deformable_group());
        return true;
    }

    namespace cum_sum_v0
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v0::CumSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::cumsum<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                               inputs[1]->get_data_ptr<T2>(),
                                               outputs[0]->get_data_ptr<T1>(),
                                               inputs[0]->get_shape(),
                                               op->is_exclusive(),
                                               op->is_reverse());
        }
    } // namespace cum_sum_v0

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::CumSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i64:
            cum_sum_v0::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: cum_sum_v0::evaluate<ET, element::Type_t::i32>(op, outputs, inputs); break;
        }
        return true;
    }

    namespace embedding_offsets_sum_v3
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v3::EmbeddingSegmentsSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::embeddingSegmentsSum<T1, T2>(
                inputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_data_ptr<T2>(),
                inputs[2]->get_data_ptr<T2>(),
                inputs.size() > 4 ? inputs[4]->get_data_ptr<T2>() : nullptr,
                inputs.size() > 5 ? inputs[5]->get_data_ptr<T1>() : nullptr,
                outputs[0]->get_data_ptr<T1>(),
                inputs[0]->get_shape(),
                inputs[1]->get_shape(),
                outputs[0]->get_shape());
        }
    } // namespace embedding_offsets_sum_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingSegmentsSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            embedding_offsets_sum_v3::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            embedding_offsets_sum_v3::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace embedding_bag_offsets_sum_v3
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v3::EmbeddingBagOffsetsSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::embeddingBagOffsetsSum<T1, T2>(
                inputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_data_ptr<T2>(),
                inputs[2]->get_data_ptr<T2>(),
                inputs.size() > 3 ? inputs[3]->get_data_ptr<T2>() : nullptr,
                inputs.size() > 4 ? inputs[4]->get_data_ptr<T1>() : nullptr,
                outputs[0]->get_data_ptr<T1>(),
                shape_size(inputs[1]->get_shape()),
                outputs[0]->get_shape());
        }
    } // namespace embedding_bag_offsets_sum_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagOffsetsSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            embedding_bag_offsets_sum_v3::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            embedding_bag_offsets_sum_v3::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace embedding_bag_packed_sum_v3
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v3::EmbeddingBagPackedSum>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::embeddingBagPackedSum<T1, T2>(
                inputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_data_ptr<T2>(),
                inputs.size() > 2 ? inputs[2]->get_data_ptr<T1>() : nullptr,
                outputs[0]->get_data_ptr<T1>(),
                inputs[1]->get_shape(),
                outputs[0]->get_shape());
        }
    } // namespace embedding_bag_packed_sum_v3

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagPackedSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            embedding_bag_packed_sum_v3::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            embedding_bag_packed_sum_v3::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }

        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::MVN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mvn<T>(inputs[0]->get_data_ptr<ET>(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_normalize_variance(),
                                   op->get_reduction_axes(),
                                   op->get_eps());
        return true;
    }

    namespace mvn_6_axes
    {
        template <typename T>
        AxisSet mvn_6_reduction_axes(const HostTensorPtr& axes_input, size_t rank)
        {
            T* a = axes_input->get_data_ptr<T>();
            auto v = std::vector<T>(a, a + axes_input->get_shape()[0]);
            std::vector<size_t> axes(v.size(), 0);
            for (size_t i = 0; i < v.size(); i++)
            {
                if (v[i] < 0)
                {
                    if (rank + v[i] < 0)
                    {
                        throw ngraph_error("Unexpected axis");
                    }
                    axes[i] = (size_t)(rank + v[i]);
                }
                else
                {
                    axes[i] = (size_t)(v[i]);
                }
            }
            return AxisSet(axes);
        }
    } // mvn_6_axes

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::MVN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        AxisSet reduction_axes;
        auto rank = inputs[0]->get_shape().size();
        if (inputs[1]->get_element_type() == element::i64)
        {
            reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int64_t>(inputs[1], rank);
        }
        else if (inputs[1]->get_element_type() == element::i32)
        {
            reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int32_t>(inputs[1], rank);
        }
        else
        {
            throw ngraph_error("Unexpected indices type");
        }
        runtime::reference::mvn_6<T>(inputs[0]->get_data_ptr<ET>(),
                                     outputs[0]->get_data_ptr<ET>(),
                                     inputs[0]->get_shape(),
                                     reduction_axes,
                                     op->get_normalize_variance(),
                                     op->get_eps(),
                                     op->get_eps_mode());
        return true;
    }

    namespace
    {
        std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape)
        {
            size_t input_size = shape_size(shape);
            std::vector<float> result(input_size);

            switch (input->get_element_type())
            {
            case element::Type_t::bf16:
            {
                bfloat16* p = input->get_data_ptr<bfloat16>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = float(p[i]);
                }
            }
            break;
            case element::Type_t::f16:
            {
                float16* p = input->get_data_ptr<float16>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = float(p[i]);
                }
            }
            break;
            case element::Type_t::f32:
            {
                float* p = input->get_data_ptr<float>();
                memcpy(result.data(), p, input_size * sizeof(float));
            }
            break;
            default: throw std::runtime_error("Unsupported data type."); break;
            }

            return result;
        }

        std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input,
                                          const Shape& shape)
        {
            size_t input_size = shape_size(shape);
            std::vector<int64_t> result(input_size);

            switch (input->get_element_type())
            {
            case element::Type_t::i8:
            {
                auto p = input->get_data_ptr<int8_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::i16:
            {
                auto p = input->get_data_ptr<int16_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::i32:
            {
                auto p = input->get_data_ptr<int32_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::i64:
            {
                auto p = input->get_data_ptr<int64_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u8:
            {
                auto p = input->get_data_ptr<uint8_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u16:
            {
                auto p = input->get_data_ptr<uint16_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u32:
            {
                auto p = input->get_data_ptr<uint32_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            case element::Type_t::u64:
            {
                auto p = input->get_data_ptr<uint64_t>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = int64_t(p[i]);
                }
            }
            break;
            default:
                throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
                break;
            }

            return result;
        }
    } // namespace

    namespace nms_v5
    {
        using V5BoxEncoding = op::v5::NonMaxSuppression::BoxEncodingType;

        struct InfoForNMS5
        {
            int64_t max_output_boxes_per_class;
            float iou_threshold;
            float score_threshold;
            float soft_nms_sigma;
            Shape out_shape;
            Shape boxes_shape;
            Shape scores_shape;
            std::vector<float> boxes_data;
            std::vector<float> scores_data;
            size_t out_shape_size;
            bool sort_result_descending;
            ngraph::element::Type output_type;
        };

        constexpr size_t boxes_port = 0;
        constexpr size_t scores_port = 1;

        PartialShape
            infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                         int64_t max_output_boxes_per_class)
        {
            const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
            const auto scores_ps = inputs[scores_port]->get_partial_shape();

            // NonMaxSuppression produces triplets
            // that have the following format: [batch_index, class_index, box_index]
            PartialShape result = {Dimension::dynamic(), 3};

            if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
            {
                const auto num_boxes_boxes = boxes_ps[1];
                if (num_boxes_boxes.is_static() && scores_ps[0].is_static() &&
                    scores_ps[1].is_static())
                {
                    const auto num_boxes = num_boxes_boxes.get_length();
                    const auto num_classes = scores_ps[1].get_length();

                    result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                                scores_ps[0].get_length();
                }
            }
            return result;
        }

        void normalize_corner(float* boxes, const Shape& boxes_shape)
        {
            size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
            for (size_t i = 0; i < total_num_of_boxes; ++i)
            {
                float* current_box = boxes + 4 * i;

                float y1 = current_box[0];
                float x1 = current_box[1];
                float y2 = current_box[2];
                float x2 = current_box[3];

                float ymin = std::min(y1, y2);
                float ymax = std::max(y1, y2);
                float xmin = std::min(x1, x2);
                float xmax = std::max(x1, x2);

                current_box[0] = ymin;
                current_box[1] = xmin;
                current_box[2] = ymax;
                current_box[3] = xmax;
            }
        }

        void normalize_center(float* boxes, const Shape& boxes_shape)
        {
            size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
            for (size_t i = 0; i < total_num_of_boxes; ++i)
            {
                float* current_box = boxes + 4 * i;

                float x_center = current_box[0];
                float y_center = current_box[1];
                float width = current_box[2];
                float height = current_box[3];

                float y1 = y_center - height / 2.0;
                float x1 = x_center - width / 2.0;
                float y2 = y_center + height / 2.0;
                float x2 = x_center + width / 2.0;

                current_box[0] = y1;
                current_box[1] = x1;
                current_box[2] = y2;
                current_box[3] = x2;
            }
        }

        void normalize_box_encoding(float* boxes,
                                    const Shape& boxes_shape,
                                    const V5BoxEncoding box_encoding)
        {
            if (box_encoding == V5BoxEncoding::CORNER)
            {
                normalize_corner(boxes, boxes_shape);
            }
            else
            {
                normalize_center(boxes, boxes_shape);
            }
        }

        std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                              const Shape& boxes_shape,
                                              const V5BoxEncoding box_encoding)
        {
            auto result = get_floats(boxes, boxes_shape);
            normalize_box_encoding(result.data(), boxes_shape, box_encoding);
            return result;
        }

        std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                               const Shape& scores_shape)
        {
            auto result = get_floats(scores, scores_shape);
            return result;
        }

        InfoForNMS5 get_info_for_nms5_eval(const std::shared_ptr<op::v5::NonMaxSuppression>& nms5,
                                           const std::vector<std::shared_ptr<HostTensor>>& inputs)
        {
            InfoForNMS5 result;

            result.max_output_boxes_per_class =
                inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
            result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
            result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
            result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

            auto selected_indices_shape =
                infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
            result.out_shape = selected_indices_shape.to_shape();

            result.boxes_shape = inputs[boxes_port]->get_shape();
            result.scores_shape = inputs[scores_port]->get_shape();

            result.boxes_data = prepare_boxes_data(
                inputs[boxes_port], result.boxes_shape, nms5->get_box_encoding());
            result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

            result.out_shape_size = shape_size(result.out_shape);

            result.sort_result_descending = nms5->get_sort_result_descending();

            result.output_type = nms5->get_output_type();

            return result;
        }

    } // namespace nms_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::NonMaxSuppression>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = nms_v5::get_info_for_nms5_eval(op, inputs);

        std::vector<int64_t> selected_indices(info.out_shape_size);
        std::vector<float> selected_scores(info.out_shape_size);
        int64_t valid_outputs = 0;

        runtime::reference::non_max_suppression(info.boxes_data.data(),
                                                info.boxes_shape,
                                                info.scores_data.data(),
                                                info.scores_shape,
                                                info.max_output_boxes_per_class,
                                                info.iou_threshold,
                                                info.score_threshold,
                                                info.soft_nms_sigma,
                                                selected_indices.data(),
                                                info.out_shape,
                                                selected_scores.data(),
                                                info.out_shape,
                                                &valid_outputs,
                                                info.sort_result_descending);

        auto selected_scores_type =
            (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

        runtime::reference::nms5_postprocessing(outputs,
                                                info.output_type,
                                                selected_indices,
                                                selected_scores,
                                                valid_outputs,
                                                selected_scores_type);
        return true;
    }

    namespace matrix_nms_v8
    {
        using SortResultType = op::v8::MatrixNms::SortResultType;
        struct InfoForNMS
        {
            Shape selected_outputs_shape;
            Shape selected_indices_shape;
            Shape boxes_shape;
            Shape scores_shape;
            std::vector<float> boxes_data;
            std::vector<float> scores_data;
            size_t selected_outputs_shape_size;
            size_t selected_indices_shape_size;
        };

        constexpr size_t boxes_port = 0;
        constexpr size_t scores_port = 1;

        PartialShape
            infer_selected_outputs_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                         int nms_top_k, int keep_top_k)
        {
            const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
            const auto scores_ps = inputs[scores_port]->get_partial_shape();

            PartialShape result = {Dimension::dynamic(), 6};

            if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
            {
                const auto num_boxes_boxes = boxes_ps[1];
                if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static())
                {
                    const auto num_boxes = num_boxes_boxes.get_length();
                    const auto num_classes = scores_ps[1].get_length();
                    int64_t max_output_boxes_per_class = 0;
                    if (nms_top_k >= 0)
                        max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
                    else
                        max_output_boxes_per_class = num_boxes;

                    auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
                    if (keep_top_k >= 0)
                        max_output_boxes_per_batch =
                            std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

                    result[0] = max_output_boxes_per_batch * scores_ps[0].get_length();
                }
            }

            return result;
        }

        std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                              const Shape& boxes_shape)
        {
            auto result = get_floats(boxes, boxes_shape);
            return result;
        }

        std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                               const Shape& scores_shape)
        {
            auto result = get_floats(scores, scores_shape);
            return result;
        }

        InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::v8::MatrixNms>& nms,
                                           const std::vector<std::shared_ptr<HostTensor>>& inputs)
        {
            InfoForNMS result;

            auto selected_outputs_shape =
                infer_selected_outputs_shape(inputs, nms->get_nms_top_k(), nms->get_keep_top_k());
            result.selected_outputs_shape = selected_outputs_shape.to_shape();
            result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

            result.boxes_shape = inputs[boxes_port]->get_shape();
            result.scores_shape = inputs[scores_port]->get_shape();

            result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
            result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

            result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
            result.selected_indices_shape_size = shape_size(result.selected_indices_shape);

            return result;
        }
    } // namespace matrix_nms_v8

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v8::MatrixNms>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = matrix_nms_v8::get_info_for_nms_eval(op, inputs);

        std::vector<float> selected_outputs(info.selected_outputs_shape_size);
        std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
        std::vector<int64_t> valid_outputs(info.boxes_shape[0]);

        runtime::reference::matrix_nms(info.boxes_data.data(),
                                                info.boxes_shape,
                                                info.scores_data.data(),
                                                info.scores_shape,
                                                op->get_attrs(),
                                                selected_outputs.data(),
                                                info.selected_outputs_shape,
                                                selected_indices.data(),
                                                info.selected_indices_shape,
                                                valid_outputs.data());

        void* pscores = nullptr;
        void* pselected_num = nullptr;
        void* prois;
        size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0));

        outputs[0]->set_shape({num_selected, 6});
        prois = outputs[0]->get_data_ptr();

        if (outputs.size() >= 2)
        {
            outputs[1]->set_shape({num_selected, 1});
            pscores = outputs[1]->get_data_ptr();
        }
        if (outputs.size() >= 3)
        {
            pselected_num = outputs[2]->get_data_ptr();
        }

        runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                pscores,
                                                pselected_num,
                                                op->get_output_type(),
                                                selected_outputs,
                                                selected_indices,
                                                valid_outputs);
        return true;
    }

    namespace multiclass_nms_v8
    {
        using SortResultType = op::v8::MulticlassNms::SortResultType;
        struct InfoForNMS
        {
            Shape selected_outputs_shape;
            Shape selected_indices_shape;
            Shape boxes_shape;
            Shape scores_shape;
            std::vector<float> boxes_data;
            std::vector<float> scores_data;
            size_t selected_outputs_shape_size;
            size_t selected_indices_shape_size;
        };

        constexpr size_t boxes_port = 0;
        constexpr size_t scores_port = 1;

        PartialShape
            infer_selected_outputs_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                         int nms_top_k, int keep_top_k)
        {
            const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
            const auto scores_ps = inputs[scores_port]->get_partial_shape();

            PartialShape result = {Dimension::dynamic(), 6};

            if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
            {
                const auto num_boxes_boxes = boxes_ps[1];
                if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static())
                {
                    const auto num_boxes = num_boxes_boxes.get_length();
                    const auto num_classes = scores_ps[1].get_length();
                    int64_t max_output_boxes_per_class = 0;
                    if (nms_top_k >= 0)
                        max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
                    else
                        max_output_boxes_per_class = num_boxes;

                    auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
                    if (keep_top_k >= 0)
                        max_output_boxes_per_batch =
                            std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

                    result[0] = max_output_boxes_per_batch * scores_ps[0].get_length();
                }
            }

            return result;
        }

        std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                              const Shape& boxes_shape)
        {
            auto result = get_floats(boxes, boxes_shape);
            return result;
        }

        std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                               const Shape& scores_shape)
        {
            auto result = get_floats(scores, scores_shape);
            return result;
        }

        InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::v8::MulticlassNms>& nms,
                                           const std::vector<std::shared_ptr<HostTensor>>& inputs)
        {
            InfoForNMS result;

            auto selected_outputs_shape =
                infer_selected_outputs_shape(inputs, nms->get_nms_top_k(), nms->get_keep_top_k());
            result.selected_outputs_shape = selected_outputs_shape.to_shape();
            result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

            result.boxes_shape = inputs[boxes_port]->get_shape();
            result.scores_shape = inputs[scores_port]->get_shape();

            result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
            result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

            result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
            result.selected_indices_shape_size = shape_size(result.selected_indices_shape);

            return result;
        }
    } // namespace multiclass_nms_v8

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v8::MulticlassNms>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = multiclass_nms_v8::get_info_for_nms_eval(op, inputs);

        std::vector<float> selected_outputs(info.selected_outputs_shape_size);
        std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
        std::vector<int64_t> valid_outputs(inputs[0]->get_shape()[0]);

        runtime::reference::multiclass_nms(info.boxes_data.data(),
                                                info.boxes_shape,
                                                info.scores_data.data(),
                                                info.scores_shape,
                                                op->get_attrs(),
                                                selected_outputs.data(),
                                                info.selected_outputs_shape,
                                                selected_indices.data(),
                                                info.selected_indices_shape,
                                                valid_outputs.data());                                                  

        void* pscores = nullptr;
        void* pselected_num = nullptr;
        void* prois;
        size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0));

        outputs[0]->set_shape({num_selected, 6});
        prois = outputs[0]->get_data_ptr();

        if (outputs.size() >= 2)
        {
            outputs[1]->set_shape({num_selected, 1});
            pscores = outputs[1]->get_data_ptr();
        }
        if (outputs.size() >= 3)
        {
            pselected_num = outputs[2]->get_data_ptr();
        }

        runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                pscores,
                                                pselected_num,
                                                op->get_output_type(),
                                                selected_outputs,
                                                selected_indices,
                                                valid_outputs);

        return true;
    }

    namespace experimental_prior_grid
    {
        struct InfoForEDPriorGrid
        {
            Shape output_shape;
            int64_t grid_h;
            int64_t grid_w;
            float stride_h;
            float stride_w;
        };

        constexpr size_t priors_port = 0;
        constexpr size_t feature_map_port = 1;

        PartialShape infer_output_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                        bool flatten)
        {
            PartialShape out_shape = {
                Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 4};

            if (flatten)
            {
                out_shape = PartialShape{Dimension::dynamic(), 4};
            }

            const auto priors_shape = inputs[priors_port]->get_partial_shape();
            const auto feature_map_shape = inputs[feature_map_port]->get_partial_shape();

            if (priors_shape.rank().is_dynamic() || feature_map_shape.rank().is_dynamic())
            {
                return out_shape;
            }

            auto num_priors = priors_shape[0];
            auto featmap_height = feature_map_shape[2];
            auto featmap_width = feature_map_shape[3];

            if (flatten)
            {
                out_shape = PartialShape{featmap_height * featmap_width * num_priors, 4};
            }
            else
            {
                out_shape = PartialShape{featmap_height, featmap_width, num_priors, 4};
            }

            return out_shape;
        }

        InfoForEDPriorGrid get_info_for_ed_prior_grid_eval(
            const std::shared_ptr<op::v6::ExperimentalDetectronPriorGridGenerator>& prior_grid,
            const std::vector<std::shared_ptr<HostTensor>>& inputs)
        {
            InfoForEDPriorGrid result;

            auto attrs = prior_grid->get_attrs();

            result.grid_h = attrs.h;
            result.grid_w = attrs.w;
            result.stride_h = attrs.stride_y;
            result.stride_w = attrs.stride_x;

            auto output_rois_shape = infer_output_shape(inputs, attrs.flatten);
            result.output_shape = output_rois_shape.to_shape();

            return result;
        }
    } // namespace experimental_prior_grid

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::ExperimentalDetectronPriorGridGenerator>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = experimental_prior_grid::get_info_for_ed_prior_grid_eval(op, inputs);

        using T = typename element_type_traits<ET>::value_type;
        outputs[0]->set_shape(info.output_shape);
        runtime::reference::experimental_detectron_prior_grid_generator<T>(
            inputs[0]->get_data_ptr<const T>(),
            inputs[0]->get_shape(),
            inputs[1]->get_shape(),
            inputs[2]->get_shape(),
            outputs[0]->get_data_ptr<T>(),
            info.grid_h,
            info.grid_w,
            info.stride_h,
            info.stride_w);

        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::ExperimentalDetectronDetectionOutput>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto attrs = op->get_attrs();
        size_t rois_num = attrs.max_detections_per_image;

        const Shape output_boxes_shape = Shape{rois_num, 4};
        const Shape output_classes_shape = Shape{rois_num};
        const Shape output_scores_shape = Shape{rois_num};

        const auto output_type = op->get_input_element_type(0);

        const auto boxes_data = get_floats(inputs[0], inputs[0]->get_shape());
        const auto input_deltas_data = get_floats(inputs[1], inputs[1]->get_shape());
        const auto input_scores_data = get_floats(inputs[2], inputs[2]->get_shape());
        const auto input_im_info_data = get_floats(inputs[3], inputs[3]->get_shape());

        std::vector<float> output_boxes(shape_size(output_boxes_shape));
        std::vector<int32_t> output_classes(shape_size(output_classes_shape));
        std::vector<float> output_scores(shape_size(output_scores_shape));

        outputs[0]->set_element_type(output_type);
        outputs[0]->set_shape(output_boxes_shape);
        outputs[1]->set_element_type(element::Type_t::i32);
        outputs[1]->set_shape(output_classes_shape);
        outputs[2]->set_element_type(output_type);
        outputs[2]->set_shape(output_scores_shape);

        runtime::reference::experimental_detectron_detection_output(boxes_data.data(),
                                                                    input_deltas_data.data(),
                                                                    input_scores_data.data(),
                                                                    input_im_info_data.data(),
                                                                    attrs,
                                                                    output_boxes.data(),
                                                                    output_scores.data(),
                                                                    output_classes.data());

        runtime::reference::experimental_detectron_detection_output_postprocessing(
            outputs[0]->get_data_ptr(),
            outputs[1]->get_data_ptr(),
            outputs[2]->get_data_ptr(),
            output_type,
            output_boxes,
            output_classes,
            output_scores,
            output_boxes_shape,
            output_classes_shape,
            output_scores_shape);

        return true;
    }

    namespace fft_v7
    {
        struct InfoForFFT7
        {
            std::vector<float> input_data;
            std::vector<int64_t> axes_data;
            Shape input_data_shape;
            Shape axes_data_shape;
            Shape output_shape;
        };

        std::vector<int64_t> get_signal_size(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                             size_t num_of_axes)
        {
            if (inputs.size() == 3)
            {
                return get_integers(inputs[2], inputs[2]->get_shape());
            }

            return std::vector<int64_t>(num_of_axes, static_cast<int64_t>(-1));
        }

        InfoForFFT7 get_info_for_fft7_eval(const std::vector<std::shared_ptr<HostTensor>>& inputs)
        {
            InfoForFFT7 result;

            result.input_data_shape = inputs[0]->get_shape();
            result.axes_data_shape = inputs[1]->get_shape();
            result.input_data = get_floats(inputs[0], result.input_data_shape);
            result.axes_data = get_integers(inputs[1], result.axes_data_shape);

            auto output_shape = result.input_data_shape;

            int64_t input_rank = static_cast<int64_t>(result.input_data_shape.size());
            int64_t complex_data_rank = input_rank - 1;
            auto canonicalized_axes = runtime::reference::canonicalize_axes(
                result.axes_data.data(), result.axes_data_shape, complex_data_rank);

            size_t num_of_axes = result.axes_data.size();
            auto signal_size = get_signal_size(inputs, num_of_axes);

            for (size_t i = 0; i < num_of_axes; ++i)
            {
                int64_t current_axis = canonicalized_axes[i];
                int64_t current_signal_size = signal_size[i];
                if (current_signal_size != -1)
                {
                    output_shape[current_axis] = current_signal_size;
                }
            }

            result.output_shape = output_shape;

            return result;
        }
    } // namespace fft_v7

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v7::DFT>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = fft_v7::get_info_for_fft7_eval(inputs);
        outputs[0]->set_shape(info.output_shape);

        std::vector<float> fft_result(shape_size(info.output_shape), 0.0f);
        runtime::reference::fft(info.input_data.data(),
                                info.input_data_shape,
                                info.axes_data.data(),
                                info.axes_data_shape,
                                fft_result.data(),
                                info.output_shape,
                                runtime::reference::FFTKind::Forward);

        const auto output_type = op->get_input_element_type(0);
        runtime::reference::fft_postprocessing(outputs, output_type, fft_result);
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v7::IDFT>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        auto info = fft_v7::get_info_for_fft7_eval(inputs);
        outputs[0]->set_shape(info.output_shape);

        std::vector<float> fft_result(shape_size(info.output_shape), 0.0f);
        runtime::reference::fft(info.input_data.data(),
                                info.input_data_shape,
                                info.axes_data.data(),
                                info.axes_data_shape,
                                fft_result.data(),
                                info.output_shape,
                                runtime::reference::FFTKind::Inverse);

        const auto output_type = op->get_input_element_type(0);
        runtime::reference::fft_postprocessing(outputs, output_type, fft_result);
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::LRN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lrn<T>(inputs[0]->get_data_ptr<ET>(),
                                   op->get_reduction_axes(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_alpha(),
                                   op->get_beta(),
                                   op->get_bias(),
                                   op->get_nsize());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::GRN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::grn<T>(inputs[0]->get_data_ptr<ET>(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   op->get_bias(),
                                   inputs[0]->get_shape());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::DetectionOutput>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                                  op->get_input_shape(0),
                                                                  op->get_input_shape(2),
                                                                  op->get_output_shape(0));
        if (op->get_input_size() == 3)
        {
            refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                          inputs[1]->get_data_ptr<const T>(),
                          inputs[2]->get_data_ptr<const T>(),
                          nullptr,
                          nullptr,
                          outputs[0]->get_data_ptr<T>());
        }
        else if (op->get_input_size() == 5)
        {
            refDetOut.run(inputs[0]->get_data_ptr<const T>(),
                          inputs[1]->get_data_ptr<const T>(),
                          inputs[2]->get_data_ptr<const T>(),
                          inputs[3]->get_data_ptr<const T>(),
                          inputs[4]->get_data_ptr<const T>(),
                          outputs[0]->get_data_ptr<T>());
        }
        else
        {
            throw ngraph_error("DetectionOutput layer supports only 3 or 5 inputs");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::ScatterNDUpdate>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        auto idxType = op->get_input_element_type(1);
        if (idxType == element::i32)
        {
            runtime::reference::scatterNdUpdate<T, int32_t>(
                inputs[0]->get_data_ptr<const T>(),
                inputs[1]->get_data_ptr<const int32_t>(),
                inputs[2]->get_data_ptr<const T>(),
                outputs[0]->get_data_ptr<T>(),
                op->get_input_shape(0),
                op->get_input_shape(1),
                op->get_input_shape(2));
        }
        else if (idxType == element::i64)
        {
            runtime::reference::scatterNdUpdate<T, int64_t>(
                inputs[0]->get_data_ptr<const T>(),
                inputs[1]->get_data_ptr<const int64_t>(),
                inputs[2]->get_data_ptr<const T>(),
                outputs[0]->get_data_ptr<T>(),
                op->get_input_shape(0),
                op->get_input_shape(1),
                op->get_input_shape(2));
        }
        else
        {
            throw ngraph_error(
                "ScatterNDUpdate layer support only i32 and i64 'indices' input precision!");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Select>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::select<T>(inputs[0]->get_data_ptr<const char>(),
                                      inputs[1]->get_data_ptr<const T>(),
                                      inputs[2]->get_data_ptr<const T>(),
                                      outputs[0]->get_data_ptr<T>(),
                                      op->get_input_shape(0),
                                      op->get_input_shape(1),
                                      op->get_input_shape(2),
                                      op->get_auto_broadcast());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::AvgPool>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::avg_pool<T>(inputs[0]->get_data_ptr<T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        inputs[0]->get_shape(),
                                        op->get_output_shape(0),
                                        op->get_kernel(),
                                        op->get_strides(),
                                        op->get_pads_begin(),
                                        op->get_pads_end(),
                                        !op->get_exclude_pad());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::HardSigmoid>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::hard_sigmoid<T>(inputs[0]->get_data_ptr<T>(),
                                            inputs[1]->get_data_ptr<const T>()[0],
                                            inputs[2]->get_data_ptr<const T>()[0],
                                            outputs[0]->get_data_ptr<T>(),
                                            shape_size(outputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Elu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::elu<T>(inputs[0]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   shape_size(inputs[0]->get_shape()),
                                   op->get_alpha());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::PriorBox>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::prior_box<T>(inputs[0]->get_data_ptr<T>(),
                                         inputs[1]->get_data_ptr<T>(),
                                         outputs[0]->get_data_ptr<float>(),
                                         outputs[0]->get_shape(),
                                         op->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Proposal>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::proposal_v0<T>(inputs[0]->get_data_ptr<T>(),
                                           inputs[1]->get_data_ptr<T>(),
                                           inputs[2]->get_data_ptr<T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           inputs[0]->get_shape(),
                                           inputs[1]->get_shape(),
                                           inputs[2]->get_shape(),
                                           outputs[0]->get_shape(),
                                           op.get()->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::Proposal>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::proposal_v4<T>(inputs[0]->get_data_ptr<T>(),
                                           inputs[1]->get_data_ptr<T>(),
                                           inputs[2]->get_data_ptr<T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           outputs[1]->get_data_ptr<T>(),
                                           inputs[0]->get_shape(),
                                           inputs[1]->get_shape(),
                                           inputs[2]->get_shape(),
                                           outputs[0]->get_shape(),
                                           outputs[1]->get_shape(),
                                           op.get()->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Mod>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mod<T>(inputs[0]->get_data_ptr<T>(),
                                   inputs[1]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   inputs[0]->get_shape(),
                                   inputs[1]->get_shape(),
                                   op->get_autob());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Selu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::selu<T>(inputs[0]->get_data_ptr<T>(),
                                    inputs[1]->get_data_ptr<T>(),
                                    inputs[2]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()),
                                    shape_size(inputs[1]->get_shape()),
                                    shape_size(inputs[2]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Ceiling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ceiling<T>(inputs[0]->get_data_ptr<T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Gelu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gelu<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    op::GeluApproximationMode::ERF,
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v7::Gelu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gelu<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    op->get_approximation_mode(),
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Relu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::relu<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Sign>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::sign<T>(inputs[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(inputs[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Abs>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::abs<T>(inputs[0]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   shape_size(inputs[0]->get_shape()));
        return true;
    }

    namespace ctc_loss_v4
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v4::CTCLoss>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::CTCLoss<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<T2>(),
                                                inputs[2]->get_data_ptr<T2>(),
                                                inputs[3]->get_data_ptr<T2>(),
                                                inputs[4]->get_data_ptr<T2>(),
                                                op->get_preprocess_collapse_repeated(),
                                                op->get_ctc_merge_repeated(),
                                                op->get_unique(),
                                                outputs[0]->get_data_ptr<T1>());
        }
    } // namespace ctc_loss_v4

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::CTCLoss>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32:
            ctc_loss_v4::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            ctc_loss_v4::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::BatchNormInference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::batch_norm_inference<T>(op->get_eps_value(),
                                                    inputs[2]->get_data_ptr<T>(),
                                                    inputs[0]->get_data_ptr<T>(),
                                                    inputs[1]->get_data_ptr<T>(),
                                                    inputs[3]->get_data_ptr<T>(),
                                                    inputs[4]->get_data_ptr<T>(),
                                                    outputs[0]->get_data_ptr<T>(),
                                                    inputs[2]->get_shape());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::BatchNormInference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::batch_norm_inference<T>(op->get_eps_value(),
                                                    inputs[0]->get_data_ptr<const T>(),
                                                    inputs[1]->get_data_ptr<const T>(),
                                                    inputs[2]->get_data_ptr<const T>(),
                                                    inputs[3]->get_data_ptr<const T>(),
                                                    inputs[4]->get_data_ptr<const T>(),
                                                    outputs[0]->get_data_ptr<T>(),
                                                    op->get_input_shape(0));
        return true;
    }

    namespace reverse_sequence_v0
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::reverse_sequence<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                         outputs[0]->get_data_ptr<T1>(),
                                                         inputs[0]->get_shape(),
                                                         op->get_batch_axis(),
                                                         op->get_sequence_axis(),
                                                         inputs[1]->get_data_ptr<T2>());
        }
    } // namespace reverse_sequence_v0

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::boolean:
            reverse_sequence_v0::evaluate<ET, element::Type_t::boolean>(op, outputs, inputs);
            break;
        case element::Type_t::i8:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i8>(op, outputs, inputs);
            break;
        case element::Type_t::i16:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i16>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            reverse_sequence_v0::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::u8:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u8>(op, outputs, inputs);
            break;
        case element::Type_t::u16:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u16>(op, outputs, inputs);
            break;
        case element::Type_t::u32:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u32>(op, outputs, inputs);
            break;
        case element::Type_t::u64:
            reverse_sequence_v0::evaluate<ET, element::Type_t::u64>(op, outputs, inputs);
            break;
        case element::Type_t::f16:
            reverse_sequence_v0::evaluate<ET, element::Type_t::f16>(op, outputs, inputs);
            break;
        case element::Type_t::f32:
            reverse_sequence_v0::evaluate<ET, element::Type_t::f32>(op, outputs, inputs);
            break;
        case element::Type_t::f64:
            reverse_sequence_v0::evaluate<ET, element::Type_t::f64>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::ExtractImagePatches>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::extract_image_patches<T>(op,
                                                     inputs[0]->get_data_ptr<T>(),
                                                     outputs[0]->get_data_ptr<T>(),
                                                     inputs[0]->get_shape(),
                                                     outputs[0]->get_shape());
        return true;
    }

    namespace convert_like_v1
    {
        template <element::Type_t ti, element::Type_t to>
        inline void evaluate(const shared_ptr<op::v1::ConvertLike>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using TI = typename element_type_traits<ti>::value_type;
            using TO = typename element_type_traits<to>::value_type;
            runtime::reference::convert<TI, TO>(inputs[0]->get_data_ptr<TI>(),
                                                outputs[0]->get_data_ptr<TO>(),
                                                shape_size(inputs[0]->get_shape()));
        }

    } // namespace convert_like_v1

    template <element::Type_t OUT_ET>
    bool evaluate(const shared_ptr<op::v1::ConvertLike>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[0]->get_element_type())
        {
        case element::Type_t::boolean:
            convert_like_v1::evaluate<element::Type_t::boolean, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u8:
            convert_like_v1::evaluate<element::Type_t::u8, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u16:
            convert_like_v1::evaluate<element::Type_t::u16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u32:
            convert_like_v1::evaluate<element::Type_t::u32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::u64:
            convert_like_v1::evaluate<element::Type_t::u64, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i8:
            convert_like_v1::evaluate<element::Type_t::i8, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i16:
            convert_like_v1::evaluate<element::Type_t::i16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
            convert_like_v1::evaluate<element::Type_t::i32, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::i64:
            convert_like_v1::evaluate<element::Type_t::i64, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::bf16:
            convert_like_v1::evaluate<element::Type_t::bf16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f16:
            convert_like_v1::evaluate<element::Type_t::f16, OUT_ET>(op, outputs, inputs);
            break;
        case element::Type_t::f32:
            convert_like_v1::evaluate<element::Type_t::f32, OUT_ET>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::RNNCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::rnn_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                        inputs[0]->get_shape(),
                                        inputs[1]->get_data_ptr<ET>(),
                                        inputs[1]->get_shape(),
                                        inputs[2]->get_data_ptr<ET>(),
                                        inputs[2]->get_shape(),
                                        inputs[3]->get_data_ptr<ET>(),
                                        inputs[3]->get_shape(),
                                        inputs[4]->get_data_ptr<ET>(),
                                        inputs[4]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        op->get_activations().front(),
                                        op->get_clip());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::LSTMCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lstm_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                         inputs[0]->get_shape(),
                                         inputs[1]->get_data_ptr<ET>(),
                                         inputs[1]->get_shape(),
                                         inputs[2]->get_data_ptr<ET>(),
                                         inputs[2]->get_shape(),
                                         inputs[3]->get_data_ptr<ET>(),
                                         inputs[3]->get_shape(),
                                         inputs[4]->get_data_ptr<ET>(),
                                         inputs[4]->get_shape(),
                                         inputs[5]->get_data_ptr<ET>(),
                                         inputs[5]->get_shape(),
                                         outputs[0]->get_data_ptr<ET>(),
                                         outputs[1]->get_data_ptr<ET>(),
                                         op->get_activations()[0],
                                         op->get_activations()[1],
                                         op->get_activations()[2],
                                         op->get_clip());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::GRUCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                        inputs[0]->get_shape(),
                                        inputs[1]->get_data_ptr<ET>(),
                                        inputs[1]->get_shape(),
                                        inputs[2]->get_data_ptr<ET>(),
                                        inputs[2]->get_shape(),
                                        inputs[3]->get_data_ptr<ET>(),
                                        inputs[3]->get_shape(),
                                        inputs[4]->get_data_ptr<ET>(),
                                        inputs[4]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        op->get_activations()[0],
                                        op->get_activations()[1],
                                        op->get_clip(),
                                        op->get_linear_before_reset());
        return true;
    }

    namespace rnn_seq_v5
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v5::RNNSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::rnn_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_clip(),
                                                     op->get_direction());
        }
    } // namespace rnn_seq_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::RNNSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[2]->get_element_type())
        {
        case element::Type_t::i64:
        case element::Type_t::u64:
            rnn_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            rnn_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace lstm_seq_v5
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::lstm_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                      inputs[0]->get_shape(),
                                                      inputs[1]->get_data_ptr<char>(),
                                                      inputs[1]->get_shape(),
                                                      inputs[2]->get_data_ptr<char>(),
                                                      inputs[2]->get_shape(),
                                                      inputs[3]->get_data_ptr<char>(),
                                                      inputs[3]->get_shape(),
                                                      inputs[4]->get_data_ptr<char>(),
                                                      inputs[4]->get_shape(),
                                                      inputs[5]->get_data_ptr<char>(),
                                                      inputs[5]->get_shape(),
                                                      inputs[6]->get_data_ptr<char>(),
                                                      inputs[6]->get_shape(),
                                                      outputs[0]->get_data_ptr<char>(),
                                                      outputs[1]->get_data_ptr<char>(),
                                                      outputs[2]->get_data_ptr<char>(),
                                                      op->get_activations()[0],
                                                      op->get_activations()[1],
                                                      op->get_activations()[2],
                                                      op->get_clip(),
                                                      op->get_direction());
        }
    } // namespace lstm_seq_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[3]->get_element_type())
        {
        case element::Type_t::i64:
        case element::Type_t::u64:
            lstm_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            lstm_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }

    namespace ti_v0
    {
        runtime::reference::custom_evaluate_function evaluate =
            [](const std::shared_ptr<ngraph::Function>& function,
               const HostTensorVector& inputs,
               HostTensorVector& outputs) -> void {
            const auto& parameters = function->get_parameters();
            const auto& parametersNumber = parameters.size();
            const auto& inputsNumber = inputs.size();
            NGRAPH_CHECK(parametersNumber == inputsNumber,
                         "Got function (",
                         function->get_friendly_name(),
                         ") with ",
                         parametersNumber,
                         " parameters, but ",
                         inputsNumber,
                         " input blobs");

            auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
            for (const auto& parameter : parameters)
            {
                const auto& parameterIndex = function->get_parameter_index(parameter);
                const auto& parameterShape = parameter->get_shape();
                const auto& parameterType = parameter->get_element_type();
                const auto& parameterSize = shape_size(parameterShape) * parameterType.size();

                const auto& input = inputs[parameterIndex];
                const auto& inputSize = input->get_size_in_bytes();
                NGRAPH_CHECK(parameterSize == inputSize,
                             "Got parameter (",
                             parameter->get_friendly_name(),
                             ") of size ",
                             parameterSize,
                             " bytes, but corresponding input with index ",
                             parameterIndex,
                             " has ",
                             inputSize,
                             " bytes");

                auto tensor = std::make_shared<runtime::HostTensor>(parameterType, parameterShape);
                tensor->write(input->get_data_ptr(), parameterSize);
                inputTensors.push_back(tensor);
            }

            const auto& results = function->get_results();
            std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputTensors;
            outputTensors.reserve(results.size());
            for (size_t i = 0; i < results.size(); ++i)
            {
                outputTensors.push_back(std::make_shared<HostTensor>());
            }
            runtime::Backend::set_backend_shared_library_search_directory("");
            auto backend = runtime::Backend::create("INTERPRETER");
            auto handle = backend->compile(function);
            handle->call_with_validate(outputTensors, inputTensors);

            outputs.reserve(outputTensors.size());
            for (const auto& tensor : outputTensors)
            {
                auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
                outputs.push_back(host_tensor);
            }
        };
    } // namespace ti_v0

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::TensorIterator>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        runtime::reference::tensor_iterator(op->get_num_iterations(),
                                            op->get_function(),
                                            op->get_output_descriptions(),
                                            op->get_input_descriptions(),
                                            outputs,
                                            inputs,
                                            ti_v0::evaluate);
        return true;
    }

    namespace gru_seq_v5
    {
        template <element::Type_t t1, element::Type_t t2>
        inline void evaluate(const shared_ptr<op::v5::GRUSequence>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using T1 = typename element_type_traits<t1>::value_type;
            using T2 = typename element_type_traits<t2>::value_type;
            runtime::reference::gru_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
                                                     inputs[0]->get_shape(),
                                                     inputs[1]->get_data_ptr<char>(),
                                                     inputs[1]->get_shape(),
                                                     inputs[2]->get_data_ptr<char>(),
                                                     inputs[2]->get_shape(),
                                                     inputs[3]->get_data_ptr<char>(),
                                                     inputs[3]->get_shape(),
                                                     inputs[4]->get_data_ptr<char>(),
                                                     inputs[4]->get_shape(),
                                                     inputs[5]->get_data_ptr<char>(),
                                                     inputs[5]->get_shape(),
                                                     outputs[0]->get_data_ptr<char>(),
                                                     outputs[1]->get_data_ptr<char>(),
                                                     op->get_activations()[0],
                                                     op->get_activations()[1],
                                                     op->get_clip(),
                                                     op->get_direction(),
                                                     op->get_linear_before_reset());
        }
    } // namespace gru_seq_v5

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::GRUSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        switch (inputs[2]->get_element_type())
        {
        case element::Type_t::i64:
        case element::Type_t::u64:
            gru_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            gru_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
            break;
        default: return false;
        }
        return true;
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ROIPooling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::roi_pooling<T>(inputs[0]->get_data_ptr<const T>(),
                                           inputs[1]->get_data_ptr<const T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           op->get_input_shape(0),
                                           op->get_input_shape(1),
                                           op->get_output_shape(0),
                                           op->get_spatial_scale(),
                                           op->get_method());
        return true;
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ReorgYolo>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        runtime::reference::reorg_yolo(inputs[0]->get_data_ptr<char>(),
                                       outputs[0]->get_data_ptr<char>(),
                                       inputs[0]->get_shape(),
                                       op->get_strides().at(0),
                                       inputs[0]->get_element_type().size());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::RegionYolo>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::region_yolo<T>(inputs[0]->get_data_ptr<const T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           inputs[0]->get_shape(),
                                           op->get_num_coords(),
                                           op->get_num_classes(),
                                           op->get_num_regions(),
                                           op->get_do_softmax(),
                                           op->get_mask());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Pad>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        runtime::reference::pad(inputs[0]->get_data_ptr<char>(),
                                inputs[1]->get_data_ptr<char>(),
                                outputs[0]->get_data_ptr<char>(),
                                shape_size(inputs[0]->get_shape()),
                                inputs[1]->get_shape(),
                                outputs[0]->get_shape(),
                                op->get_pads_end(),
                                op->get_pads_begin(),
                                op->get_pad_mode());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GatherTree>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        runtime::reference::gather_tree(inputs[0]->get_data_ptr<const char>(),
                                        inputs[1]->get_data_ptr<const char>(),
                                        inputs[2]->get_data_ptr<const char>(),
                                        inputs[3]->get_data_ptr<const char>(),
                                        outputs[0]->get_data_ptr<char>(),
                                        op->get_input_shape(0),
                                        op->get_input_shape(1),
                                        op->get_input_shape(2),
                                        op->get_input_shape(3),
                                        inputs[1]->get_element_type());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::FakeQuantize>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::fake_quantize<T>(inputs[0]->get_data_ptr<const T>(),
                                             inputs[1]->get_data_ptr<const T>(),
                                             inputs[2]->get_data_ptr<const T>(),
                                             inputs[3]->get_data_ptr<const T>(),
                                             inputs[4]->get_data_ptr<const T>(),
                                             outputs[0]->get_data_ptr<T>(),
                                             op->get_input_shape(0),
                                             op->get_input_shape(1),
                                             op->get_input_shape(2),
                                             op->get_input_shape(3),
                                             op->get_input_shape(4),
                                             op->get_levels());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::NormalizeL2>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::normalize_l2<T>(inputs[0]->get_data_ptr<const T>(),
                                            outputs[0]->get_data_ptr<T>(),
                                            op->get_input_shape(0),
                                            op->get_reduction_axes(),
                                            op->get_eps(),
                                            op->get_eps_mode());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::CTCGreedyDecoder>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ctc_greedy_decoder<T>(inputs[0]->get_data_ptr<const T>(),
                                                  inputs[1]->get_data_ptr<const T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  outputs[0]->get_shape(),
                                                  op->get_ctc_merge_repeated());
        return true;
    }

    namespace ctc_greedy_decoder_v6
    {
        template <element::Type_t T1, element::Type_t T2, element::Type_t TOUT>
        inline void evaluate(const shared_ptr<op::v6::CTCGreedyDecoderSeqLen>& op,
                             const HostTensorVector& outputs,
                             const HostTensorVector& inputs)
        {
            using TF = typename element_type_traits<T1>::value_type;
            using TI = typename element_type_traits<T2>::value_type;
            using TIND1 = typename element_type_traits<TOUT>::value_type;
            TI blank_index_val = inputs[0]->get_shape().back() - 1;
            const TI *blank_index = &blank_index_val;
            if (inputs.size() == 3) {
                blank_index = inputs[2]->get_data_ptr<const TI>();
            }
            if (op->get_sequence_length_type() == element::i32)
            {
                runtime::reference::ctc_greedy_decoder_seq_len<TF>(
                    inputs[0]->get_data_ptr<const TF>(),
                    inputs[1]->get_data_ptr<const TI>(),
                    blank_index,
                    outputs[0]->get_data_ptr<TIND1>(),
                    outputs[1]->get_data_ptr<int32_t>(),
                    inputs[0]->get_shape(),
                    outputs[0]->get_shape(),
                    op->get_merge_repeated());
            }
            else if (op->get_sequence_length_type() == element::i64)
            {
                runtime::reference::ctc_greedy_decoder_seq_len<TF>(
                    inputs[0]->get_data_ptr<const TF>(),
                    inputs[1]->get_data_ptr<const TI>(),
                    blank_index,
                    outputs[0]->get_data_ptr<TIND1>(),
                    outputs[1]->get_data_ptr<int64_t>(),
                    inputs[0]->get_shape(),
                    outputs[0]->get_shape(),
                    op->get_merge_repeated());
            }
        }
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::CTCGreedyDecoderSeqLen>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto& dataType = inputs[0]->get_element_type();
        const auto& seqLenType = inputs[1]->get_element_type();
        if (dataType == element::Type_t::f16 && seqLenType == element::Type_t::i32)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f16, element::Type_t::i32, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f32 && seqLenType == element::Type_t::i32)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f32, element::Type_t::i32, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f64 && seqLenType == element::Type_t::i32)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f64, element::Type_t::i32, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f16 && seqLenType == element::Type_t::i64)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f16, element::Type_t::i64, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f32 && seqLenType == element::Type_t::i64)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f32, element::Type_t::i64, ET>(
                op, outputs, inputs);
        }
        else if (dataType == element::Type_t::f64 && seqLenType == element::Type_t::i64)
        {
            ctc_greedy_decoder_v6::evaluate<element::Type_t::f64, element::Type_t::i64, ET>(
                op, outputs, inputs);
        }
        else
        {
            return false;
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::ExperimentalDetectronTopKROIs>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        size_t max_rois = op->get_max_rois();
        outputs[0]->set_shape(Shape{max_rois, 4});
        runtime::reference::experimental_detectron_topk_rois<T>(inputs[0]->get_data_ptr<const T>(),
                                                                inputs[1]->get_data_ptr<const T>(),
                                                                inputs[0]->get_shape(),
                                                                inputs[1]->get_shape(),
                                                                max_rois,
                                                                outputs[0]->get_data_ptr<T>());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto attrs = op->get_attrs();

        size_t post_nms_count = 0;
        if (attrs.post_nms_count < 0)
        {
            throw ngraph_error("The attribute post_nms_count of the operation "
                               "ExperimentalDetectronGenerateProposalsSingleImage must be a "
                               "nonnegative integer.");
        }
        else
        {
            post_nms_count = static_cast<size_t>(attrs.post_nms_count);
        }

        const Shape output_rois_shape = Shape{post_nms_count, 4};
        const Shape output_scores_shape = Shape{post_nms_count};

        const auto output_type = op->get_input_element_type(0);

        const auto im_info_shape = inputs[0]->get_shape();
        const auto anchors_shape = inputs[1]->get_shape();
        const auto deltas_shape = inputs[2]->get_shape();
        const auto scores_shape = inputs[3]->get_shape();

        const auto im_info_data = get_floats(inputs[0], im_info_shape);
        const auto anchors_data = get_floats(inputs[1], anchors_shape);
        const auto deltas_data = get_floats(inputs[2], deltas_shape);
        const auto scores_data = get_floats(inputs[3], scores_shape);

        std::vector<float> output_rois(shape_size(output_rois_shape));
        std::vector<float> output_scores(shape_size(output_scores_shape));

        outputs[0]->set_element_type(output_type);
        outputs[0]->set_shape(output_rois_shape);
        outputs[1]->set_element_type(output_type);
        outputs[1]->set_shape(output_scores_shape);

        runtime::reference::experimental_detectron_proposals_single_image(im_info_data.data(),
                                                                          anchors_data.data(),
                                                                          deltas_data.data(),
                                                                          scores_data.data(),
                                                                          attrs,
                                                                          im_info_shape,
                                                                          anchors_shape,
                                                                          deltas_shape,
                                                                          scores_shape,
                                                                          output_rois.data(),
                                                                          output_scores.data());
        runtime::reference::experimental_detectron_proposals_single_image_postprocessing(
            outputs[0]->get_data_ptr(),
            outputs[1]->get_data_ptr(),
            output_type,
            output_rois,
            output_scores,
            output_rois_shape,
            output_scores_shape);

        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::SquaredDifference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::squared_difference<T>(inputs[0]->get_data_ptr<const T>(),
                                                  inputs[1]->get_data_ptr<const T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  op->get_autob());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v6::GatherElements>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        Shape params_shape = inputs[0]->get_shape();
        Shape indices_shape = inputs[1]->get_shape();

        outputs[0]->set_shape(indices_shape);

        if (inputs[1]->get_element_type() == element::i64)
        {
            runtime::reference::gather_elements<T, int64_t>(inputs[0]->get_data_ptr<ET>(),
                                                            inputs[1]->get_data_ptr<int64_t>(),
                                                            outputs[0]->get_data_ptr<ET>(),
                                                            inputs[0]->get_shape(),
                                                            inputs[1]->get_shape(),
                                                            outputs[0]->get_shape(),
                                                            op->get_axis());
        }
        else if (inputs[1]->get_element_type() == element::i32)
        {
            runtime::reference::gather_elements<T, int32_t>(inputs[0]->get_data_ptr<ET>(),
                                                            inputs[1]->get_data_ptr<int32_t>(),
                                                            outputs[0]->get_data_ptr<ET>(),
                                                            inputs[0]->get_shape(),
                                                            inputs[1]->get_shape(),
                                                            outputs[0]->get_shape(),
                                                            op->get_axis());
        }
        else
        {
            throw ngraph_error("Unexpected indices type");
        }

        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::GatherND>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        if (op->get_input_element_type(1) == element::i64)
        {
            runtime::reference::gather_nd<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                                      inputs[1]->get_data_ptr<int64_t>(),
                                                      outputs[0]->get_data_ptr<T>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_output_shape(0),
                                                      op->get_batch_dims());
        }
        else if (op->get_input_element_type(1) == element::i32)
        {
            runtime::reference::gather_nd<T, int32_t>(inputs[0]->get_data_ptr<T>(),
                                                      inputs[1]->get_data_ptr<int32_t>(),
                                                      outputs[0]->get_data_ptr<T>(),
                                                      op->get_input_shape(0),
                                                      op->get_input_shape(1),
                                                      op->get_output_shape(0),
                                                      op->get_batch_dims());
        }
        else
        {
            throw ngraph_error("Unexpected indices type for GatherND operation");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::LogSoftmax>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        int64_t i_axis = op->get_axis();
        if (i_axis < 0)
        {
            i_axis += inputs[0]->get_partial_shape().rank().get_length();
        }
        runtime::reference::log_softmax<T>(inputs[0]->get_data_ptr<const T>(),
                                           outputs[0]->get_data_ptr<T>(),
                                           op->get_output_shape(0),
                                           AxisSet{(size_t)i_axis});
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::PSROIPooling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                             inputs[0]->get_shape(),
                                             inputs[1]->get_data_ptr<T>(),
                                             inputs[1]->get_shape(),
                                             outputs[0]->get_data_ptr<T>(),
                                             outputs[0]->get_shape(),
                                             op->get_mode(),
                                             op->get_spatial_scale(),
                                             op->get_spatial_bins_x(),
                                             op->get_spatial_bins_y());

        return true;
    }
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::DeformablePSROIPooling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        NGRAPH_CHECK(inputs.size() > 1 && inputs[1]->get_shape().size() == 2,
                        "2D tensor must be provided as second input. ");
        outputs[0]->set_shape({inputs[1]->get_shape()[0],
                               static_cast<size_t>(op->get_output_dim()),
                               static_cast<size_t>(op->get_group_size()),
                               static_cast<size_t>(op->get_group_size())});

        const bool has_offset_intput = inputs.size() == 3;
        if (has_offset_intput)
        {
            runtime::reference::deformable_psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<T>(),
                                                inputs[1]->get_shape(),
                                                inputs[2]->get_data_ptr<T>(),
                                                inputs[2]->get_shape(),
                                                outputs[0]->get_data_ptr<T>(),
                                                outputs[0]->get_shape(),
                                                op->get_mode(),
                                                op->get_spatial_scale(),
                                                op->get_spatial_bins_x(),
                                                op->get_spatial_bins_y(),
                                                op->get_trans_std(),
                                                op->get_part_size());
        }
        else
        {
           runtime::reference::deformable_psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<T>(),
                                                inputs[1]->get_shape(),
                                                nullptr,
                                                ngraph::Shape(),
                                                outputs[0]->get_data_ptr<T>(),
                                                outputs[0]->get_shape(),
                                                op->get_mode(),
                                                op->get_spatial_scale(),
                                                op->get_spatial_bins_x(),
                                                op->get_spatial_bins_y(),
                                                op->get_trans_std(),
                                                op->get_part_size());
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v7::Roll>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto& shiftType = inputs[1]->get_element_type();
        std::vector<int64_t> shift_int64;
        if (shiftType == element::Type_t::i32)
        {
            auto shift = inputs[1]->get_data_ptr<const int32_t>();
            shift_int64.resize(shape_size(inputs[1]->get_shape()));
            std::transform(shift,
                           shift + shape_size(inputs[1]->get_shape()),
                           shift_int64.begin(),
                           [](const int32_t& elem) { return static_cast<int64_t>(elem); });
        }
        const auto& axesType = inputs[2]->get_element_type();
        std::vector<int64_t> axes_int64;
        if (axesType == element::Type_t::i32)
        {
            auto axes = inputs[2]->get_data_ptr<const int32_t>();
            axes_int64.resize(shape_size(inputs[2]->get_shape()));
            std::transform(axes,
                           axes + shape_size(inputs[2]->get_shape()),
                           axes_int64.begin(),
                           [](const int32_t& elem) { return static_cast<int64_t>(elem); });
        }
        runtime::reference::roll(inputs[0]->get_data_ptr<const char>(),
                                 inputs[1]->get_element_type() != element::Type_t::i64
                                     ? shift_int64.data()
                                     : inputs[1]->get_data_ptr<const int64_t>(),
                                 inputs[2]->get_element_type() != element::Type_t::i64
                                     ? axes_int64.data()
                                     : inputs[2]->get_data_ptr<const int64_t>(),
                                 outputs[0]->get_data_ptr<char>(),
                                 inputs[0]->get_shape(),
                                 inputs[1]->get_shape(),
                                 inputs[2]->get_shape(),
                                 inputs[0]->get_element_type().size());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v7::Einsum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto equation = op->get_equation();
        runtime::reference::einsum(outputs, inputs, equation);
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v8::AdaptiveAvgPool>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::adaptive_avg_pool(inputs[0]->get_data_ptr<T>(),
                                              outputs[0]->get_data_ptr<T>(),
                                              inputs[0]->get_shape(),
                                              op->get_output_shape(0));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v8::AdaptiveMaxPool>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        if (op->get_index_element_type() == element::i32) {
            runtime::reference::adaptive_max_pool(inputs[0]->get_data_ptr<T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  outputs[1]->get_data_ptr<int32_t>(),
                                                  inputs[0]->get_shape(),
                                                  op->get_output_shape(0));
        } else if (op->get_index_element_type() == element::i64) {
            runtime::reference::adaptive_max_pool(inputs[0]->get_data_ptr<T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  outputs[1]->get_data_ptr<int64_t>(),
                                                  inputs[0]->get_shape(),
                                                  op->get_output_shape(0));
        }
        return true;
    }

    template <typename T>
    bool evaluate_node(std::shared_ptr<Node> node,
                       const HostTensorVector& outputs,
                       const HostTensorVector& inputs)
    {
        auto element_type = node->get_output_element_type(0);
        if (is_type<op::v1::Select>(node))
        {
            element_type = node->get_input_element_type(1);
        }
        else if (is_type<op::v0::PriorBox>(node))
        {
            element_type = node->get_input_element_type(0);
        }
        for (size_t i = 1; i < node->outputs().size(); i++)
        {
            if ((is_type<op::v5::NonMaxSuppression>(node) ||
                 is_type<op::v8::MulticlassNms>(node) ||
                 is_type<op::v8::MatrixNms>(node) ||
                 is_type<op::v6::ExperimentalDetectronDetectionOutput>(node) ||
                 is_type<op::v8::AdaptiveMaxPool>(node)) &&
                 i == 1)
            {
                continue;
            }
        }
        switch (element_type)
        {
        case element::Type_t::boolean:
            return evaluate<element::Type_t::boolean>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::bf16:
            return evaluate<element::Type_t::bf16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f16:
            return evaluate<element::Type_t::f16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f64:
            return evaluate<element::Type_t::f64>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f32:
            return evaluate<element::Type_t::f32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i8:
            return evaluate<element::Type_t::i8>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i16:
            return evaluate<element::Type_t::i16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i32:
            return evaluate<element::Type_t::i32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i64:
            return evaluate<element::Type_t::i64>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u8:
            return evaluate<element::Type_t::u8>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u16:
            return evaluate<element::Type_t::u16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u32:
            return evaluate<element::Type_t::u32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u64:
            return evaluate<element::Type_t::u64>(as_type_ptr<T>(node), outputs, inputs);
        default:
            throw ngraph_error(std::string("Unhandled data type ") +
                               node->get_element_type().get_type_name() +
                               std::string("in evaluate_node()"));
        }
    }
} // namespace

runtime::interpreter::EvaluatorsMap& runtime::interpreter::get_evaluators_map()
{
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef NGRAPH_OP
    };
    return evaluatorsMap;
}
