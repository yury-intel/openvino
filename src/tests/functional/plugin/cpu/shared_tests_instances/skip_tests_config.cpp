// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <ie_system_conf.h>

#include <string>
#include <vector>

#include "ie_parallel.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // TODO: Issue 31841
        R"(.*(QuantGroupConvBackpropData3D).*)",
        // TODO: Issue 31843
        R"(.*(QuantConvBackpropData3D).*)",
        R"(.*(QuantConvBackpropData2D).*(QG=Perchannel).*)",
        R"(.*(QuantGroupConvBackpropData2D).*(QG=Perchannel).*)",
        // TODO: Issue 33886
        R"(.*(QuantGroupConv2D).*)",
        R"(.*(QuantGroupConv3D).*)",
        // TODO: Issue: 34518
        R"(.*RangeLayerTest.*)",
        R"(.*(RangeAddSubgraphTest).*Start=1.2.*Stop=(5.2|-5.2).*Step=(0.1|-0.1).*netPRC=FP16.*)",
        R"(.*(RangeNumpyAddSubgraphTest).*netPRC=FP16.*)",
        // TODO: Issue: 43793
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*iPRC=0.*_iLT=1.*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*_oLT=1.*)",
        // TODO: Issue: 63469
        R"(.*ConversionLayerTest.*ConvertLike.*)",
        // TODO: Issue: 34055
        R"(.*ShapeOfLayerTest.*)",
        R"(.*ReluShapeOfSubgraphTest.*)",
        // TODO: Issue: 43314
        R"(.*Broadcast.*mode=BIDIRECTIONAL.*inNPrec=BOOL.*)",
        // TODO: Issue 43417 sporadic issue, looks like an issue in test, reproducible only on Windows platform
        R"(.*decomposition1_batch=5_hidden_size=10_input_size=30_.*tanh.relu.*_clip=0_linear_before_reset=1.*_targetDevice=CPU_.*)",
        // Skip platforms that do not support BF16 (i.e. sse, avx, avx2)
        R"(.*(BF|bf)16.*(jit_avx(?!5)|jit_sse|ref).*)",
        // TODO: Incorrect blob sizes for node BinaryConvolution_X
        R"(.*BinaryConvolutionLayerTest.*)",
        R"(.*ClampLayerTest.*netPrc=(I64|I32).*)",
        R"(.*ClampLayerTest.*netPrc=U64.*)",
        // TODO: 53618. BF16 gemm ncsp convolution crash
        R"(.*_GroupConv.*_inFmts=nc.*_primitive=jit_gemm.*ENFORCE_BF16=YES.*)",
        // TODO: 53578. fork DW bf16 convolution does not support 3d cases yet
        R"(.*_DW_GroupConv.*_inFmts=(ndhwc|nCdhw16c).*ENFORCE_BF16=YES.*)",
        // TODO: 56143. Enable nspc convolutions for bf16 precision
        R"(.*ConvolutionLayerCPUTest.*_inFmts=(ndhwc|nhwc).*ENFORCE_BF16=YES.*)",
        // TODO: 56827. Sporadic test failures
        R"(.*smoke_Conv.+_FP32.ConvolutionLayerCPUTest\.CompareWithRefs.*TS=\(\(.\.67.+\).*inFmts=n.+c.*_primitive=jit_avx2.*)",
        // incorrect jit_uni_planar_convolution with dilation = {1, 2, 1} and output channel 1
        R"(.*smoke_Convolution3D.*D=\(1.2.1\)_O=1.*)",

        // TODO: Issue: 35627. CPU Normalize supports from 2D to 4D blobs
        R"(.*NormalizeL2_1D.*)",
        R"(.*NormalizeL2_5D.*)",
        // Issue: 59788. mkldnn_normalize_nchw applies eps after sqrt for across_spatial
        R"(.*NormalizeL2_.*axes=\(1.2.*_eps=100.*)",
        R"(.*NormalizeL2_.*axes=\(2.1.*_eps=100.*)",
        R"(.*NormalizeL2_.*axes=\(3.1.2.*_eps=100.*)",

        // Unsupported operation of type: NormalizeL2 name : Doesn't support reduction axes: (2.2)
        R"(.*BF16NetworkRestore1.*)",
        R"(.*MobileNet_ssd_with_branching.*)",

        // TODO: 57562 No dynamic output shape support
        R"(.*NonZeroLayerTest.*)",
        // TODO: 69084 Not constant Axis input produces dynamic output shape.
        R"(.*GatherLayerTestCPU.*constAx=False.*)",
        // TODO: 74961.  Enforce precision via inType and outType does not work properly.
        R"(.*(RNN|GRU|LSTM).*ENFORCE_BF16=YES.*)",
        // Not expected behavior
        R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*layout=(95|OIHW).*)",
        R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*layout=(95|OIHW).*)",
        R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetOutBlobWithDifferentLayouts.*layout=HW.*)",
        R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetInBlobWithDifferentLayouts.*layout=NHWC.*targetDevice=(AUTO|MULTI).*)",
        R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetOutBlobWithDifferentLayouts.*layout=CN.*targetDevice=(AUTO|MULTI).*)",
        R"(.*Behavior.*InferRequestSetBlobByType.*Batched.*)",
        R"(.*Auto_Behavior.*InferRequestIOBBlobTest.*canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*)",
        R"(.*Auto.*Behavior.*ExecutableNetworkBaseTest.*canLoadCorrectNetworkToGetExecutableWithIncorrectConfig.*)",
        R"(.*(Auto|Multi).*Behavior.*CorrectConfigAPITests.*CanSetExclusiveAsyncRequests.*)",
        R"(.*(Auto|Multi).*Behavior.*IncorrectConfigTests.*CanNotLoadNetworkWithIncorrectConfig.*)",
        R"(.*OVExecutableNetworkBaseTest.*(CanGetInputsInfoAndCheck|CanSetConfigToExecNet).*)",
        R"(.*Behavior.*CorrectConfigCheck.*(canSetConfigAndCheckGetConfig|canSetConfigTwiceAndCheckGetConfig).*CPU_BIND_THREAD=YES.*)",
        // Issue: 62846 Here shape[1] is not the channel dimension size
        R"(.*smoke_Reduce.*KeepNoDims.*(_axes=\((0.*|.*1.*)).*Fused=.*PerChannel.*)",
        // Issue: 72021 Unreasonable abs_threshold for comparing bf16 results
        R"(.*smoke_Reduce.*type=(Prod|Min).*netPRC=(BF|bf)16.*)",
        // TODO: 56520 Accuracy mismatch
        R"(.*ReduceOpsLayerTest.*type=Mean_.*netPRC=(I64|I32).*)",
        R"(.*ReduceOpsLayerTest.*type=Mean_.*netPRC=U64.*)",
        // Not implemented yet:
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
        R"(.*(Auto|Multi).*Behavior.*ExecutableNetworkBaseTest.*checkGetExecGraphInfo.*)",
        R"(.*(Auto|Multi).*Behavior.*ExecutableNetworkBaseTest.*CanCreateTwoExeNetworksAndCheckFunction.*)",
        R"(.*(Auto|Multi).*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution).*)",
        R"(.*(Auto|Multi).*Behavior.*ExecutableNetworkBaseTest.*CheckExecGraphInfoSerialization.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNetWithIncorrectConfig.*)",
        R"(.*Hetero.*Behavior.*ExecutableNetworkBaseTest.*ExecGraphInfo.*)",
        R"(.*Hetero.*Behavior.*ExecutableNetworkBaseTest.*CanCreateTwoExeNetworksAndCheckFunction.*)",

        // CPU plugin does not support some precisions
        R"(smoke_CachingSupportCase_CPU/LoadNetworkCacheTestBase.CompareWithRefImpl/ReadConcatSplitAssign_f32_batch1_CPU)",

        // CPU does not support dynamic rank
        // Issue: CVS-66778
        R"(.*smoke_BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
        R"(.*smoke_Hetero_BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
        R"(.*smoke_Auto_BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
        R"(.*smoke_BehaviorTests.*DynamicOutputToDynamicInput.*)",
        R"(.*smoke_BehaviorTests.*DynamicInputToDynamicOutput.*)",
        R"(.*smoke_Hetero_BehaviorTests.*DynamicOutputToDynamicInput.*)",
        R"(.*smoke_Hetero_BehaviorTests.*DynamicInputToDynamicOutput.*)",
        R"(.*smoke_Auto_BehaviorTests.*DynamicOutputToDynamicInput.*)",
        R"(.*smoke_Auto_BehaviorTests.*DynamicInputToDynamicOutput.*)",

        // TODO: Issue CVS-51680
        R"(.*BehaviorTests.*canRun3SyncRequestsConsistentlyFromThreads.*CPU_THROUGHPUT.*)",
        // Issue 67214
        R"(smoke_PrePostProcess.*resize_and_convert_layout_i8.*)",
        // Issue 67910
        R"(.*smoke_PrePostProcess.*two_inputs_trivial.*)",
        // TODO: CVS-67255
        R"(smoke_If.*SimpleIf2OutTest.*)",

        // Issue: 69086
        // need to add support convert BIN -> FP32
        // if we set output precision as BIN, when we create output blob precision looks like UNSPECIFIED
        R"(.*smoke_FakeQuantizeLayerCPUTest.*bin.*)",
        // Issue: 69088
        // bad accuracy
        R"(.*smoke_FakeQuantizeLayerCPUTest_Decompos.
            *IS=_TS=\(\(4\.5\.6\.7\)\)_RS=\(\(1\.1\.6\.1\)\)_\(\(1\.5\.6\.1\)\)_\(\(1\.1\.1\.1\)\)_\(\(1\.1\.6\.1\)\).*)",
        // Issue: 69222
        R"(.*smoke_PriorBoxClustered.*PriorBoxClusteredLayerCPUTest.*_netPRC=f16_.*)",
        // TODO : CVS-69533
        R"(.*ConvolutionLayerCPUTest.*IS=\{.+\}.*_Fused=.*Add\(Parameters\).*)",
        R"(.*GroupConvolutionLayerCPUTest.*IS=\{.+\}.*_Fused=.*Add\(Parameters\).*)",
        // // Issue: 74817
        // // Sporadic failings with NAN on Dynamic shape cases with jit implementation
        // R"(.*DefConvLayoutTest7.*)",
        // Issue: 71968
        R"(.*LSTMSequenceCommonZeroClip.*PURE.*CONST.*hidden_size=10.*sigmoid.sigmoid.sigmoid.*reverse.*FP32_targetDevice=CPU.*)",
        // Issue: 72005
        // there are some inconsistency between cpu plugin and ng ref
        // for ctcMergeRepeated is true when legal randomized inputs value.
        // Failure happened on win and macos for current seeds.
        R"(.*CTCLossLayerTest.*CMR=1.*)",
        R"(.*CTCLossLayerCPUTest.*ctcMergeRepeated=1.*)",
        // Issue: 71756
        R"(.*Deconv_.*D_(Blocked|DW|1x1)_.*DeconvolutionLayerCPUTest\.CompareWithRefs.*inFmts=(nChw16c|nCdhw16c)_outFmts=(nChw16c|nCdhw16c)_primitive=jit_avx512_.*Fused=Multiply\(PerChannel\)\.Add\(PerChannel\).*)",
        R"(.*smoke_GroupDeconv_(2|3)D_Blocked_BF16.*S=(\(2\.2\)|\(2\.2\.2\))_PB=(\(0\.0\)|\(0\.0\.0\))_PE=(\(0\.0\)|\(0\.0\.0\))_D=(\(1\.1\)|\(1\.1\.1\))_.*_O=64_G=4.*)",
        // Issue: 59594
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*BOOL.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*MIXED.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*Q78.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*U4.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*I4.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*BIN.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*CUSTOM.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*UNSPECIFIED.*)",
        // Issue:
        R"(.*smoke_VariadicSplit4D_CPU_zero_dims.*)",
        // New API tensor tests
        R"(.*OVInferRequestCheckTensorPrecision.*type=i4.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u1.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u4.*)",
        // Issue: 75022
        R"(.*OVExecutableNetworkBaseTest.*LoadNetworkToDefaultDeviceNoThrow.*)",
        R"(.*IEClassBasicTest.*LoadNetworkToDefaultDeviceNoThrow.*)",
        // Issue: 77390
        R"(.*LoopLayerCPUTest.*exec_cond=0.*)",
        R"(.*LoopLayerCPUTest.*trip_count=0.*)",
        R"(.*LoopForDiffShapesLayerCPUTest.*exec_cond=0.*)",
        R"(.*LoopForDiffShapesLayerCPUTest.*trip_count=0.*)",
    };

#define FIX_62820 0
#if FIX_62820 && ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
    retVector.emplace_back(R"(.*ReusableCPUStreamsExecutor.*)");
#endif

#ifdef __APPLE__
    // TODO: Issue 55717
    // retVector.emplace_back(R"(.*smoke_LPT.*ReduceMinTransformation.*f32.*)");
#endif
    if (!InferenceEngine::with_cpu_x86_avx512_core()) {
        // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
        // tests are useless on such platforms
        retVector.emplace_back(R"(.*(BF|bf)16.*)");
        retVector.emplace_back(R"(.*bfloat16.*)");
    }

    return retVector;
}
