# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import pytest

# test.BACKEND_NAME is a configuration variable determining which
# nGraph backend tests will use. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more details.
BACKEND_NAME = None

# test.MODEL_ZOO_DIR is a configuration variable providing the path
# to the ZOO of ONNX models to test. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more
# details.
MODEL_ZOO_DIR = None

# test.MODEL_ZOO_XFAIL is a configuration variable which enable xfails for model zoo.
MODEL_ZOO_XFAIL = False


def xfail_test(reason="Mark the test as expected to fail", strict=True):
    return pytest.mark.xfail(reason=reason, strict=strict)


skip_segfault = pytest.mark.skip(reason="Segmentation fault error")
xfail_issue_33488 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "MaxUnpool")
xfail_issue_33512 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Einsum")
xfail_issue_33515 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "BitShift")
xfail_issue_33535 = xfail_test(reason="nGraph does not support the following ONNX operations:"
                                      "DynamicQuantizeLinear")
xfail_issue_33538 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Scan")
xfail_issue_33540 = xfail_test(reason="RuntimeError: GRUCell operation has a form that is not supported "
                                      "GRUCell_<number> should be converted to GRUCellIE operation")
xfail_issue_33589 = xfail_test(reason="nGraph does not support the following ONNX operations:"
                                      "IsNaN and isInf")
xfail_issue_33595 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Unique")
xfail_issue_33596 = xfail_test(reason="RuntimeError: nGraph does not support different sequence operations:"
                                      "ConcatFromSequence, SequenceConstruct, SequenceAt, SplitToSequence,"
                                      "SequenceEmpty, SequenceInsert, SequenceErase, SequenceLength ")
xfail_issue_33606 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Det")
xfail_issue_33616 = xfail_test(reason="Add ceil_mode for Max and Avg pooling (reference implementation)")
xfail_issue_33644 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Compress")
xfail_issue_33651 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "TfIdfVectorizer")
xfail_issue_34310 = xfail_test(reason="RuntimeError: Error of validate layer: LSTMSequence_<number> with "
                                      "type: LSTMSequence. Layer is not instance of RNNLayer class")
xfail_issue_34314 = xfail_test(reason="RuntimeError: RNNCell operation has a form that is not "
                               "supported.RNNCell_<number> should be converted to RNNCellIE operation")
xfail_issue_34323 = xfail_test(reason="RuntimeError: data [value] doesn't exist")
xfail_issue_34327 = xfail_test(reason="RuntimeError: '<value>' layer has different "
                                      "IN and OUT channels number")
xfail_issue_33581 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "GatherElements")
xfail_issue_35911 = xfail_test(reason="Assertion error: Pad model mismatch error")
xfail_issue_35912 = xfail_test(reason="RuntimeError: Error of validate layer: B with type: "
                                      "Pad. Cannot parse parameter pads_end  from IR for layer B. "
                                      "Value -1,0 cannot be casted to int.")
xfail_issue_35914 = xfail_test(reason="IndexError: too many indices for array: "
                                      "array is 0-dimensional, but 1 were indexed")
xfail_issue_35915 = xfail_test(reason="RuntimeError: Eltwise node with unsupported combination "
                                      "of input and output types")
xfail_issue_35916 = xfail_test(reason="RuntimeError: Unsupported input dims count for layer Z")
xfail_issue_35917 = xfail_test(reason="RuntimeError: Unsupported input dims count for "
                                      "layer MatMul")
xfail_issue_35918 = xfail_test(reason="onnx.onnx_cpp2py_export.checker.ValidationError: "
                                      "Mismatched attribute type in 'test_node : alpha'")
xfail_issue_35921 = xfail_test(reason="ValueError - shapes mismatch in gemm")
xfail_issue_35923 = xfail_test(reason="RuntimeError: PReLU without weights is not supported")
xfail_issue_35924 = xfail_test(reason="Assertion error - elu results mismatch")
xfail_issue_35925 = xfail_test(reason="Assertion error - reduction ops results mismatch")
xfail_issue_35926 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format I64 is "
                                      "not supported yet...")
xfail_issue_35927 = xfail_test(reason="RuntimeError: B has zero dimension that is not allowable")
xfail_issue_35929 = xfail_test(reason="RuntimeError: Incorrect precision f64!")
xfail_issue_35930 = xfail_test(reason="onnx.onnx_cpp2py_export.checker.ValidationError: "
                                      "Required attribute 'to' is missing.")
xfail_issue_35932 = xfail_test(reason="Assertion error - logsoftmax results mismatch")
xfail_issue_36437 = xfail_test(reason="RuntimeError: Cannot find blob with name: <value>")
xfail_issue_36476 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format U32 is "
                               "not supported yet...")
xfail_issue_36478 = xfail_test(reason="RuntimeError: [NOT_IMPLEMENTED] Input image format U64 is "
                               "not supported yet...")
xfail_issue_36479 = xfail_test(reason="Assertion error - basic computation on ndarrays mismatch")
xfail_issue_36480 = xfail_test(reason="RuntimeError: [NOT_FOUND] Unsupported property dummy_option "
                               "by CPU plugin")
xfail_issue_36483 = xfail_test(reason="RuntimeError: Unsupported primitive of type: "
                               "Ceiling name: <value>")
xfail_issue_36485 = xfail_test(reason="RuntimeError: Check 'm_group >= 1' failed at "
                               "/openvino/ngraph/core/src/op/shuffle_channels.cpp:77:")
xfail_issue_36486 = xfail_test(reason="RuntimeError: HardSigmoid operation should be converted "
                                      "to HardSigmoid_IE")
xfail_issue_36487 = xfail_test(reason="Assertion error - mvn operator computation mismatch")
xfail_issue_38084 = xfail_test(reason="RuntimeError: AssertionFailed: layer->get_output_partial_shape(i)"
                                      "is_static() nGraph <value> operation with name: <value> cannot be"
                                      "converted to <value> layer with name: <value> because output"
                                      "with index 0 contains dynamic shapes: {<value>}. Try to use "
                                      "CNNNetwork::reshape() method in order to specialize shapes "
                                      "before the conversion.")
xfail_issue_38085 = xfail_test(reason="RuntimeError: Interpolate operation should be converted to Interp")
xfail_issue_38086 = xfail_test(reason="RuntimeError: Quantize layer input '<value>' doesn't have blobs")
xfail_issue_38087 = xfail_test(reason="RuntimeError: Cannot cast to tensor desc. Format is unsupported!")
xfail_issue_38088 = xfail_test(reason="RuntimeError: Check '((axis >= axis_range_min) && "
                                      "(axis <= axis_range_max))' failed at "
                                      "/openvino/ngraph/core/src/validation_util.cpp:913: "
                                      "Split Parameter axis <value> out of the tensor rank range <value>.")
xfail_issue_38089 = xfail_test(reason="RuntimeError: Node 2 contains empty child edge for index 0")
xfail_issue_38090 = xfail_test(reason="AssertionError: Items types are not equal")
xfail_issue_38091 = xfail_test(reason="AssertionError: Mismatched elements")
xfail_issue_38699 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Gradient")
xfail_issue_38701 = xfail_test(reason="RuntimeError: unsupported element type: STRING")
xfail_issue_38705 = xfail_test(reason="IndexError: deque::_M_range_check: __n (which is 0)"
                                      ">= this->size() (which is 0)")
xfail_issue_38706 = xfail_test(reason="RuntimeError: output_3.0 has zero dimension which is not allowed")
xfail_issue_38707 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "SoftmaxCrossEntropyLoss")
xfail_issue_38708 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Slice): y>': "
                                      "Axes input must be constant")
xfail_issue_38710 = xfail_test(reason="RuntimeError: roi has zero dimension which is not allowed")
xfail_issue_38712 = xfail_test(reason="RuntimeError: Check '(fmod == 1) "
                                      "While validating ONNX node '<Node(Mod): z>': "
                                      "Only 'fmod=1' mode is supported for mod operator.")
xfail_issue_38713 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Momentum")
xfail_issue_38714 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Resize): Y>'"
                                      "Check 'element::Type::merge(element_type, element_type,"
                                      "node->get_input_element_type(i))' "
                                      "While validating node 'v1::<name> (sizes[0]:i64{4},"
                                      "Convert_29306[0]:f32{4}) -> (dynamic?)' with friendly_name '<name>':"
                                      "Argument element types are inconsistent.")
xfail_issue_38715 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(OneHot): y>':"
                                      "While validating node 'v1::OneHot OneHot_<number>"
                                      "(Convert_13525[0]:i64{3}, depth[0]:f32{},"
                                      "Squeeze_13532[0]:i32{}, Squeeze_13529[0]:i32{}) -> (dynamic?)'"
                                      "with friendly_name 'OneHot_13534':"
                                      "Depth must be integral element type.")
xfail_issue_38717 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "GreaterOrEqual")
xfail_issue_38719 = xfail_test(reason="nGraph does not support the following ONNX operations: GatherND")
xfail_issue_38721 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Pow): z>': "
                                      "While validating node 'v1::Power Power_<number>"
                                      "(x[0]:f32{3}, y[0]:i64{3}) -> (dynamic?)' with friendly_name "
                                      "'Power_<number>': Argument element types are inconsistent.")
xfail_issue_38722 = xfail_test(reason="RuntimeError: While validating ONNX nodes MatMulInteger"
                                      "and QLinearMatMul"
                                      "Input0 scale and input0 zero point shape must be same and 1")
xfail_issue_38723 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "QLinearConv")
xfail_issue_38724 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Resize): Y>':"
                                      "tf_crop_and_resize - this type of coordinate transformation mode"
                                      "is not supported. Choose one of the following modes:"
                                      "tf_half_pixel_for_nn, asymmetric, align_corners, pytorch_half_pixel,"
                                      "half_pixel")
xfail_issue_38725 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Loop):"
                                      "value info has no element type specified")
xfail_issue_38726 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "LessOrEqual")
xfail_issue_38732 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ConvInteger")
xfail_issue_38733 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Celu")
xfail_issue_38734 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Adam")
xfail_issue_38735 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "ai.onnx.preview.training.Adagrad")
xfail_issue_38736 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "NegativeLogLikelihoodLoss")

# Model ONNX Zoo issues:
xfail_issue_36533 = xfail_test(reason="AssertionError: zoo models results mismatch")
xfail_issue_36537 = xfail_test(reason="ngraph.exceptions.UserInputError: (Provided tensor's shape:"
                                      "%s does not match the expected: %s."
                                      "<PartialShape: <value>>, <PartialShape: <value>>)")
xfail_issue_37687 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "Loop")
xfail_issue_39682 = xfail_test(reason="model with IR version >= 3 must specify opset_import for ONNX")
xfail_issue_39683 = xfail_test(reason="convolution.W in initializer but not in graph input")
xfail_issue_39684 = xfail_test(reason="ngraph.exceptions.UserInputError:"
                                      "('Expected %s parameters, received %s.', 1, 3)")
xfail_issue_39685 = xfail_test(reason="RuntimeError: While validating node 'v1::Transpose 315,"
                                      "Constant_9353 -> (f32{?,?,?,?})' with friendly_name '315':"
                                      "Input order must have shape [n], where n is the rank of arg.")

# Model MSFT issues:
xfail_issue_36465 = xfail_test(reason="LSTM_Seq_lens: RuntimeError: get_shape was called on a "
                                      "descriptor::Tensor with dynamic shape")
xfail_issue_37957 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations:"
                                      "com.microsoft.CropAndResize, com.microsoft.GatherND,"
                                      "com.microsoft.Pad, com.microsoft.Range")
xfail_issue_39669 = xfail_test(reason="AssertionError: This model has no test data")
