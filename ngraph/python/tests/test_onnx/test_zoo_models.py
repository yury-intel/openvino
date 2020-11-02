# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
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
import tests
from operator import itemgetter
from pathlib import Path
import os

from tests.test_onnx.utils import OpenVinoOnnxBackend
from tests.test_onnx.utils.model_importer import ModelImportRunner

from tests import (
    xfail_issue_38701,
    xfail_issue_39682,
    xfail_issue_37687,
    xfail_issue_39683,
    xfail_issue_36533,
    xfail_issue_39684,
    xfail_issue_35926,
    xfail_issue_36537,
    xfail_issue_39685,
    xfail_issue_37957,
    xfail_issue_36465,
    xfail_issue_38090,
    xfail_issue_38084,
    xfail_issue_39669,
    xfail_issue_38726)

MODELS_ROOT_DIR = tests.MODEL_ZOO_DIR

tolerance_map = {
    "arcface_lresnet100e_opset8": {"atol": 0.001, "rtol": 0.001},
    "fp16_inception_v1": {"atol": 0.001, "rtol": 0.001},
    "mobilenet_opset7": {"atol": 0.001, "rtol": 0.001},
    "resnet50_v2_opset7": {"atol": 0.001, "rtol": 0.001},
    "test_mobilenetv2-1.0": {"atol": 0.001, "rtol": 0.001},
    "test_resnet101v2": {"atol": 0.001, "rtol": 0.001},
    "test_resnet18v2": {"atol": 0.001, "rtol": 0.001},
    "test_resnet34v2": {"atol": 0.001, "rtol": 0.001},
    "test_resnet50v2": {"atol": 0.001, "rtol": 0.001},
    "mosaic": {"atol": 0.001, "rtol": 0.001},
    "pointilism": {"atol": 0.001, "rtol": 0.001},
    "rain_princess": {"atol": 0.001, "rtol": 0.001},
    "udnie": {"atol": 0.001, "rtol": 0.001},
    "candy": {"atol": 0.003, "rtol": 0.003},
    "densenet-3": {"atol": 1e-7, "rtol": 0.0011},
    "arcfaceresnet100-8": {"atol": 0.001, "rtol": 0.001},
    "mobilenetv2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet101-v1-7": {"atol": 0.001, "rtol": 0.001},
    "resnet101-v2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet152-v1-7": {"atol": 1e-7, "rtol": 0.003},
    "resnet152-v2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet18-v1-7": {"atol": 0.001, "rtol": 0.001},
    "resnet18-v2-7": {"atol": 0.001, "rtol": 0.001},
    "resnet34-v2-7": {"atol": 0.001, "rtol": 0.001},
    "vgg16-7": {"atol": 0.001, "rtol": 0.001},
    "vgg19-bn-7": {"atol": 0.001, "rtol": 0.001},
    "tinyyolov2-7": {"atol": 0.001, "rtol": 0.001},
    "tinyyolov2-8": {"atol": 0.001, "rtol": 0.001},
    "candy-8": {"atol": 0.001, "rtol": 0.001},
    "candy-9": {"atol": 0.007, "rtol": 0.001},
    "mosaic-8": {"atol": 0.003, "rtol": 0.001},
    "mosaic-9": {"atol": 0.001, "rtol": 0.001},
    "pointilism-8": {"atol": 0.001, "rtol": 0.001},
    "pointilism-9": {"atol": 0.001, "rtol": 0.001},
    "rain-princess-8": {"atol": 0.001, "rtol": 0.001},
    "rain-princess-9": {"atol": 0.001, "rtol": 0.001},
    "udnie-8": {"atol": 0.001, "rtol": 0.001},
    "udnie-9": {"atol": 0.001, "rtol": 0.001},
}

zoo_models = []
# rglob doesn't work for symlinks, so models have to be physically somwhere inside "MODELS_ROOT_DIR"
for path in Path(MODELS_ROOT_DIR).rglob("*.onnx"):
    mdir = path.parent
    file_name = path.name
    if path.is_file() and not file_name.startswith("."):
        model = {"model_name": path, "model_file": file_name, "dir": mdir}
        basedir = mdir.stem
        if basedir in tolerance_map:
            # updated model looks now:
            # {"model_name": path, "model_file": file, "dir": mdir, "atol": ..., "rtol": ...}
            model.update(tolerance_map[basedir])
        zoo_models.append(model)

if len(zoo_models) > 0:
    sorted(zoo_models, key=itemgetter("model_name"))

    # Set backend device name to be used instead of hardcoded by ONNX BackendTest class ones.
    OpenVinoOnnxBackend.backend_name = tests.BACKEND_NAME

    # import all test cases at global scope to make them visible to pytest
    backend_test = ModelImportRunner(OpenVinoOnnxBackend, zoo_models, __name__, MODELS_ROOT_DIR)
    test_cases = backend_test.test_cases["OnnxBackendModelImportTest"]
    # flake8: noqa: E501
    if tests.MODEL_ZOO_XFAIL:
        import_xfail_list = [
            # ONNX Model Zoo
            (xfail_issue_38701, "test_onnx_model_zoo_text_machine_comprehension_bidirectional_attention_flow_model_bidaf_9_bidaf_bidaf_cpu"),
            (xfail_issue_39682, "test_onnx_model_zoo_vision_classification_mnist_model_mnist_1_mnist_model_cpu"),
            (xfail_issue_37687, "test_onnx_model_zoo_vision_object_detection_segmentation_ssd_mobilenetv1_model_ssd_mobilenet_v1_10_ssd_mobilenet_v1_ssd_mobilenet_v1_cpu"),
            (xfail_issue_37687, "test_onnx_model_zoo_vision_object_detection_segmentation_yolov3_model_yolov3_10_yolov3_yolov3_cpu"),
            (xfail_issue_37687, "test_onnx_model_zoo_vision_object_detection_segmentation_tiny_yolov3_model_tiny_yolov3_11_yolov3_tiny_cpu"),
            (xfail_issue_39683, "test_onnx_model_zoo_vision_object_detection_segmentation_tiny_yolov2_model_tinyyolov2_1_tiny_yolov2_model_cpu"),
            (xfail_issue_38726, "test_onnx_model_zoo_text_machine_comprehension_t5_model_t5_decoder_with_lm_head_12_t5_decoder_with_lm_head_cpu"),

            # Model MSFT
            (xfail_issue_37687, "test_MSFT_opset10_mlperf_ssd_mobilenet_300_ssd_mobilenet_v1_coco_2018_01_28_cpu"),
            (xfail_issue_37687, "test_MSFT_opset10_mlperf_ssd_resnet34_1200_ssd_resnet34_mAP_20.2_cpu"),
            (xfail_issue_37687, "test_MSFT_opset10_yolov3_yolov3_cpu"),
            (xfail_issue_37687, "test_MSFT_opset11_tinyyolov3_yolov3_tiny_cpu"),
            (xfail_issue_37687, "test_MSFT_opset10_mlperf_ssd_resnet34_1200_ssd_resnet34_mAP_20.2_cpu"),
            (xfail_issue_37957, "test_MSFT_opset10_mask_rcnn_keras_mask_rcnn_keras_cpu"),
            (xfail_issue_36465, "test_MSFT_opset9_LSTM_Seq_lens_unpacked_model_cpu"),
        ]
        for test_case in import_xfail_list:
            xfail, test_name = test_case
            xfail(getattr(test_cases, test_name))

    del test_cases

    test_cases = backend_test.test_cases["OnnxBackendModelExecutionTest"]
    if tests.MODEL_ZOO_XFAIL:
        execution_xfail_list = [
            # ONNX Model Zoo
            (xfail_issue_36533, "test_onnx_model_zoo_vision_object_detection_segmentation_duc_model_ResNet101_DUC_7_ResNet101_DUC_HDC_ResNet101_DUC_HDC_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_object_detection_segmentation_retinanet_model_retinanet_9_test_retinanet_resnet101_retinanet_9_cpu"),
            (xfail_issue_39684, "test_onnx_model_zoo_vision_object_detection_segmentation_yolov4_model_yolov4_yolov4_yolov4_cpu"),
            (xfail_issue_35926, "test_onnx_model_zoo_text_machine_comprehension_bert_squad_model_bertsquad_10_download_sample_10_bertsquad10_cpu"),
            (xfail_issue_35926, "test_onnx_model_zoo_text_machine_comprehension_gpt_2_model_gpt2_10_GPT2_model_cpu"),
            (xfail_issue_35926, "test_onnx_model_zoo_text_machine_comprehension_roberta_model_roberta_base_11_roberta_base_11_roberta_base_11_cpu"),
            (xfail_issue_35926, "test_onnx_model_zoo_text_machine_comprehension_bert_squad_model_bertsquad_8_download_sample_8_bertsquad8_cpu"),
            (xfail_issue_35926, "test_onnx_model_zoo_text_machine_comprehension_gpt_2_model_gpt2_lm_head_10_GPT_2_LM_HEAD_model_cpu"),
            (xfail_issue_36537, "test_onnx_model_zoo_vision_classification_shufflenet_model_shufflenet_v2_10_model_test_shufflenetv2_model_cpu"),
            (xfail_issue_36537, "test_onnx_model_zoo_vision_classification_efficientnet_lite4_model_efficientnet_lite4_11_efficientnet_lite4_efficientnet_lite4_cpu"),
            (xfail_issue_39685, "test_onnx_model_zoo_text_machine_comprehension_roberta_model_roberta_sequence_classification_9_roberta_sequence_classification_9_roberta_sequence_classification_9_cpu"),
            (xfail_issue_39669, "test_onnx_model_zoo_text_machine_comprehension_t5_model_t5_encoder_12_t5_encoder_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_mobilenet_model_mobilenetv2_7_mobilenetv2_1.0_mobilenetv2_1.0_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_resnet_model_resnet101_v2_7_resnet101v2_resnet101_v2_7_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_resnet_model_resnet34_v2_7_resnet34v2_resnet34_v2_7_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_vgg_model_vgg16_7_vgg16_vgg16_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_resnet_model_resnet152_v2_7_resnet152v2_resnet152_v2_7_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_vgg_model_vgg19_bn_7_vgg19_bn_vgg19_bn_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_object_detection_segmentation_tiny_yolov2_model_tinyyolov2_7_tiny_yolov2_model_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_object_detection_segmentation_tiny_yolov2_model_tinyyolov2_8_tiny_yolov2_Model_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_style_transfer_fast_neural_style_model_mosaic_8_mosaic_mosaic_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_resnet_model_resnet18_v2_7_resnet18v2_resnet18_v2_7_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_resnet_model_resnet101_v1_7_resnet101v1_resnet101_v1_7_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_resnet_model_resnet152_v1_7_resnet152v1_resnet152_v1_7_cpu"),
            (xfail_issue_36533, "test_onnx_model_zoo_vision_classification_densenet_121_model_densenet_3_densenet121_model_cpu"),
            (xfail_issue_38084, "test_onnx_model_zoo_vision_object_detection_segmentation_mask_rcnn_model_MaskRCNN_10_mask_rcnn_R_50_FPN_1x_cpu"),
            (xfail_issue_38090, "test_onnx_model_zoo_vision_object_detection_segmentation_ssd_model_ssd_10_model_cpu"),
            (xfail_issue_38084, "test_onnx_model_zoo_vision_object_detection_segmentation_faster_rcnn_model_FasterRCNN_10_faster_rcnn_R_50_FPN_1x_cpu"),

            # Model MSFT
            (xfail_issue_36533, "test_MSFT_opset10_tf_inception_v2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset8_test_tiny_yolov2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset9_tf_inception_v2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset8_tf_inception_v2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset7_test_resnet152v2_resnet152v2_cpu"),
            (xfail_issue_36533, "test_MSFT_opset7_test_tiny_yolov2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset7_tf_inception_v2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset11_tf_inception_v2_model_cpu"),
            (xfail_issue_36533, "test_MSFT_opset7_test_mobilenetv2_1.0_mobilenetv2_1.0_cpu"),

            (xfail_issue_38090, "test_MSFT_opset7_fp16_tiny_yolov2_onnxzoo_winmlperf_tiny_yolov2_cpu"),
            (xfail_issue_38090, "test_MSFT_opset8_fp16_inception_v1_onnxzoo_lotus_inception_v1_cpu"),
            (xfail_issue_38090, "test_MSFT_opset8_fp16_tiny_yolov2_onnxzoo_winmlperf_tiny_yolov2_cpu"),
            (xfail_issue_38090, "test_MSFT_opset8_fp16_shufflenet_onnxzoo_lotus_shufflenet_cpu"),
            (xfail_issue_38090, "test_MSFT_opset7_fp16_inception_v1_onnxzoo_lotus_inception_v1_cpu"),
            (xfail_issue_38090, "test_MSFT_opset10_mlperf_resnet_resnet50_v1_cpu"),
            (xfail_issue_38090, "test_MSFT_opset7_fp16_shufflenet_onnxzoo_lotus_shufflenet_cpu"),

            (xfail_issue_38084, "test_MSFT_opset10_mask_rcnn_mask_rcnn_R_50_FPN_1x_cpu"),
            (xfail_issue_38084, "test_MSFT_opset10_faster_rcnn_faster_rcnn_R_50_FPN_1x_cpu"),

            (xfail_issue_39669, "test_MSFT_opset9_cgan_cgan_cpu"),
            (xfail_issue_35926, "test_MSFT_opset10_BERT_Squad_bertsquad10_cpu"),
        ]
        for test_case in import_xfail_list + execution_xfail_list:
            xfail, test_name = test_case
            xfail(getattr(test_cases, test_name))

    del test_cases

    globals().update(backend_test.enable_report().test_cases)
