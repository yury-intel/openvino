// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_def_conv_node.h"
#include "mkldnn_reorder_node.h"
#include "mkldnn_input_node.h"

#include "mkldnn_eltwise_node.h"
#include <string>
#include <vector>
#include <math.h>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "ie_parallel.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_def_conv_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_def_conv_kernel_f32 : public jit_uni_def_conv_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_def_conv_kernel_f32)

    explicit jit_uni_def_conv_kernel_f32(jit_def_conv_params jcp) : jit_uni_def_conv_kernel(jcp), jit_generator() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        this->preamble();

        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
        mov(reg_def_off, ptr[this->param1 + GET_OFF(off)]);
        mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
        if (jcp_.with_bias)
            mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
        if (jcp_.with_modulation)
            mov(reg_modulation, ptr[this->param1 + GET_OFF(modulation)]);
        mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
        mov(reg_input_buffer, ptr[this->param1 + GET_OFF(buf)]);
        mov(reg_oh_pos, ptr[param1 + GET_OFF(oh_pos)]);

        ow_loop();

        this->postamble();

        prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    using Ymm = const Xbyak::Ymm;
    using Xmm = const Xbyak::Xmm;
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg8_t = const Xbyak::Reg8;

    reg64_t reg_input = r8;
    reg64_t reg_def_off = r9;
    reg64_t reg_kernel = r10;
    reg64_t reg_bias = r11;
    reg64_t reg_modulation = rcx;
    reg64_t reg_output = r12;
    reg64_t reg_oh_pos = r13;
    reg64_t aux_reg_bias = rsi;
    reg64_t reg_ow_pos = rdx;
    reg64_t aux_reg_output = reg_ow_pos;
    reg64_t reg_dg_iter = reg_output;
    reg64_t aux_reg_input = rax;
    reg64_t aux2_reg_input = reg_kernel;
    reg64_t reg_ic_iter = rbx;
    reg64_t reg_oc_work = reg_ic_iter;
    reg64_t aux_reg_def_off = reg_bias;
    reg64_t aux_reg_input_buffer = r14;
    reg32_t reg_tmp_32 = r15d;
    reg64_t reg_tmp_64 = r15;
    reg64_t reg_table = rbp;
    reg64_t reg_input_buffer = aux_reg_input;
    reg64_t aux_reg_kernel = reg_table;
    reg64_t aux2_reg_kernel = reg_tmp_64;
    reg64_t aux2_reg_input_buffer = aux_reg_bias;
    reg64_t aux3_reg_input_buffer = reg_input;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    inline Xbyak::Address table_val(int index)
    { return ptr[reg_table + index * vlen]; }

    inline Vmm get_vmm_ker(int idx) { return Vmm(idx + 0); }
    inline Vmm get_vmm_src(int idx) { return Vmm(idx + 1); }
    inline Vmm get_vmm_acc(int idx) { return Vmm(idx + jcp_.ur_w + 1); }
    inline Ymm get_ymm_acc(int idx) { return Ymm(idx + jcp_.ur_w + 1); }
    inline Xmm get_xmm_acc(int idx) { return Xmm(idx + jcp_.ur_w + 1); }

    Xbyak::Label l_table;

    void ow_loop() {
        Label ow_loop_main;
        Label ow_tail;

        mov(reg_ow_pos, 0);

        L(ow_loop_main); {
            cmp(reg_ow_pos, jcp_.ow - jcp_.ur_w);
            jg(ow_tail, T_NEAR);

            oc_loop(jcp_.ur_w);

            add(reg_input, jcp_.ur_w * jcp_.stride_w * jcp_.ic * jcp_.typesize_in);
            add(reg_def_off, jcp_.ur_w * jcp_.typesize_off);
            if (jcp_.with_modulation) {
                add(reg_modulation, jcp_.ur_w * jcp_.typesize_modulation);
            }
            add(reg_output, jcp_.ur_w * jcp_.oc * jcp_.typesize_out);

            add(reg_ow_pos, jcp_.ur_w);
            jmp(ow_loop_main, T_NEAR);
        }

        L(ow_tail); {
            if (jcp_.ow % jcp_.ur_w != 0)
                oc_loop(jcp_.ow % jcp_.ur_w);
        }
    }

    void prepare_table() {
        align(64);
        L(l_table);
        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(0);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(float2int(static_cast<float>(jcp_.ih)));
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(float2int(static_cast<float>(jcp_.iw)));
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(jcp_.ih - 1);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(jcp_.iw - 1);
        }

        for (size_t d = 0; d < vlen / sizeof(int32_t); ++d) {
            dd(1);
        }
    }

    void apply_filter(int ow_step, int oc_blocks_step, int oc_step, int ic_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        for (int kh = 0; kh < jcp_.kh; kh++) {
            for (int kw = 0; kw < jcp_.kw; kw++) {
                for (int ic = 0; ic < ic_step; ic++) {
                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_src = get_vmm_src(ow);
                        size_t inp_off = (size_t) ow * jcp_.kh * jcp_.kw * jcp_.ic + kh * jcp_.kw * jcp_.ic + kw * jcp_.ic + ic;

                        uni_vbroadcastss(vmm_src, ptr[aux2_reg_input_buffer + inp_off * jcp_.typesize_in]);
                    }

                    for (int r = 0; r < repeats; r++) {
                        for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                            Vmm vmm_ker = get_vmm_ker(0);
                            size_t ker_off = (size_t) ocb * jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block +
                                             kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block +
                                             kw * jcp_.ic_block * jcp_.oc_block +
                                             ic * jcp_.oc_block + r * jcp_.oc_block / 2;

                            uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp_.typesize_in]);
                            for (int ow = 0; ow < ow_step; ow++) {
                                Vmm vmm_src = get_vmm_src(ow);
                                Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);

                                if (isa == cpu::x64::sse41 && ow > 0) {
                                    uni_vmovups(vmm_ker, ptr[aux2_reg_kernel + ker_off * jcp_.typesize_in]);
                                }

                                uni_vfmadd231ps(vmm_acc, vmm_ker, vmm_src);
                            }
                        }
                    }
                }
            }
        }
    }

    void init_accums(int ow_step, int oc_blocks_step, int oc_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;
        for (int r = 0; r < repeats; r++) {
            for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);

                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
                }
            }
        }
    }

    void ic_loop(int ow_step, int oc_blocks_step, int oc_step) {
        Label ic_main_loop;
        Label ic_tail;
        Label exit;

        push(reg_oc_work);
        if (jcp_.with_bias)
            push(aux_reg_bias);

        mov(aux2_reg_kernel, aux_reg_kernel);
        mov(aux2_reg_input_buffer, reg_input_buffer);

        mov(reg_ic_iter, jcp_.ic);

        init_accums(ow_step, oc_blocks_step, oc_step);

        L(ic_main_loop); {
            cmp(reg_ic_iter, jcp_.ic_block);
            jl(ic_tail, T_NEAR);

            apply_filter(ow_step, oc_blocks_step, oc_step, jcp_.ic_block);

            add(aux2_reg_input_buffer, jcp_.ic_block * jcp_.typesize_in);
            add(aux2_reg_kernel, jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            sub(reg_ic_iter, jcp_.ic_block);
            jmp(ic_main_loop, T_NEAR);
        }

        L(ic_tail); {
            if (jcp_.ic % jcp_.ic_block != 0) {
                apply_filter(ow_step, oc_blocks_step, oc_step, jcp_.ic % jcp_.ic_block);
            }
        }
        if (jcp_.with_bias)
            pop(aux_reg_bias);
        pop(reg_oc_work);
    }

    void interpolate_input(int ow_step) {
        Label dg_loop;
        Label dg_loop_end;

        mov(reg_table, l_table);
        mov(aux_reg_def_off, reg_def_off);
        mov(aux_reg_input, reg_input);
        mov(aux2_reg_input_buffer, aux_reg_input_buffer);
        if (jcp_.with_modulation) {
            push(reg_modulation);
        }
        xor_(reg_dg_iter, reg_dg_iter);

        const int ic_per_def_group = jcp_.ic / jcp_.dg;
        L(dg_loop); {
            cmp(reg_dg_iter, jcp_.dg);
            jge(dg_loop_end, T_NEAR);

            for (int ow = 0; ow < ow_step; ow++) {
                for (int kh = 0; kh < jcp_.kh; kh++) {
                    for (int kw = 0; kw < jcp_.kw; kw++) {
                        Label init_with_zeros;
                        Label ic_loop_main;
                        Label ic_loop_tail;
                        Label ic_loop_zeros;
                        Label loop_end;
                        Label v1_condition_end_main;
                        Label v2_condition_end_main;
                        Label v3_condition_end_main;
                        Label v4_condition_end_main;
                        Label v1_condition_end_tail;
                        Label v2_condition_end_tail;
                        Label v3_condition_end_tail;
                        Label v4_condition_end_tail;

                        mov(aux2_reg_input, aux_reg_input);
                        add(aux2_reg_input, (ow * jcp_.stride_w * jcp_.ic) * jcp_.typesize_in);

                        mov(aux3_reg_input_buffer, aux2_reg_input_buffer);
                        add(aux3_reg_input_buffer, (ow * jcp_.kh * jcp_.kw * jcp_.ic) * jcp_.typesize_in);

                        Xmm xmm_tmp = Xmm(0);

                        Xmm xmm_map_h = Xmm(2);
                        Xmm xmm_ih_in = Xmm(4);
                        Xmm xmm_ih_im = Xmm(1);
                        Xmm xmm_h_low = xmm_ih_in;
                        Xmm xmm_h_high = xmm_ih_im;
                        Xmm xmm_lh = xmm_map_h;
                        Xmm xmm_hh = Xmm(3);

                        Xmm xmm_map_w = Xmm(6);
                        Xmm xmm_iw_in = Xmm(8);
                        Xmm xmm_iw_im = Xmm(5);
                        Xmm xmm_w_low = xmm_iw_in;
                        Xmm xmm_w_high = xmm_iw_im;
                        Xmm xmm_lw = xmm_map_w;
                        Xmm xmm_hw = Xmm(7);

                        Xmm xmm_v1_off = xmm_lh;
                        Xmm xmm_v2_off = xmm_hh;
                        Xmm xmm_v3_off = xmm_lw;
                        Xmm xmm_v4_off = xmm_hw;

                        Xmm xmm_cur_height = Xmm(13);
                        Xmm xmm_cur_width = Xmm(14);

                        Xmm xmm_w1 = Xmm(9);
                        Xmm xmm_w2 = Xmm(10);
                        Xmm xmm_w3 = Xmm(11);
                        Xmm xmm_w4 = Xmm(12);

                        Xmm xmm_v1 = xmm_v1_off;
                        Xmm xmm_v2 = xmm_v2_off;
                        Xmm xmm_v3 = xmm_v3_off;
                        Xmm xmm_v4 = xmm_v4_off;

                        Vmm vmm_w1 = Vmm(xmm_w1.getIdx());
                        Vmm vmm_w2 = Vmm(xmm_w2.getIdx());
                        Vmm vmm_w3 = Vmm(xmm_w3.getIdx());
                        Vmm vmm_w4 = Vmm(xmm_w4.getIdx());

                        Vmm vmm_v1 = Vmm(xmm_v1_off.getIdx());
                        Vmm vmm_v2 = Vmm(xmm_v2_off.getIdx());
                        Vmm vmm_v3 = Vmm(xmm_v3_off.getIdx());
                        Vmm vmm_v4 = Vmm(xmm_v4_off.getIdx());

                        // condition check

                        size_t def_off_h = ((2 * (kh * jcp_.kw + kw) + 0) * jcp_.oh * jcp_.ow) + ow;
                        mov(reg_tmp_32, ptr[aux_reg_def_off + def_off_h * jcp_.typesize_off]);
                        movq(xmm_tmp, reg_tmp_64);
                        mov(reg_tmp_32, float2int(static_cast<float>((kh * (jcp_.dilate_h + 1)))));
                        movq(xmm_map_h, reg_tmp_64);
                        addss(xmm_map_h, xmm_tmp);

                        mov(reg_tmp_32, jcp_.stride_h);
                        imul(reg_tmp_32, reg_oh_pos);
                        sub(reg_tmp_32, jcp_.t_pad);
                        movq(xmm_ih_in, reg_tmp_64);

                        cvtsi2ss(xmm_ih_im, reg_tmp_32);
                        addss(xmm_ih_im, xmm_map_h);

                        if (jcp_.with_bi_pad) {
                            movss(xmm_tmp, xmm_ih_im);
                            cvtps2dq(xmm_tmp, xmm_tmp);
                            cmpss(xmm_tmp, table_val(1), 1);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            je(init_with_zeros, T_NEAR);

                            movss(xmm_tmp, xmm_ih_im);
                            cvtps2dq(xmm_tmp, xmm_tmp);
//                            paddd(xmm_tmp, table_val(5));
                            cmpss(xmm_tmp, table_val(6), 0x0e);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(init_with_zeros, T_NEAR);
                        } else {
                            movss(xmm_tmp, xmm_ih_im);
                            cmpss(xmm_tmp, table_val(0), 1);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(init_with_zeros, T_NEAR);

                            cmpss(xmm_ih_im, table_val(1), 1);
                            movq(reg_tmp_64, xmm_ih_im);
                            cmp(reg_tmp_32, 0);
                            je(init_with_zeros, T_NEAR);
                        }


                        size_t def_off_w = ((2 * (kh * jcp_.kw + kw) + 1) * jcp_.oh * jcp_.ow) + ow;
                        mov(reg_tmp_32, ptr[aux_reg_def_off + def_off_w * jcp_.typesize_off]);

                        movq(xmm_tmp, reg_tmp_64);
                        mov(reg_tmp_32, float2int(static_cast<float>((kw * (jcp_.dilate_w + 1)))));
                        movq(xmm_map_w, reg_tmp_64);
                        addss(xmm_map_w, xmm_tmp);

                        mov(reg_tmp_32, jcp_.stride_w);
                        imul(reg_tmp_32, reg_ow_pos);
                        sub(reg_tmp_32, jcp_.l_pad - ow * jcp_.stride_w);
                        movq(xmm_iw_in, reg_tmp_64);

                        cvtsi2ss(xmm_iw_im, reg_tmp_32);
                        addss(xmm_iw_im, xmm_map_w);

                        if (jcp_.with_bi_pad) {
                            movss(xmm_tmp, xmm_iw_im);
                            cvtps2dq(xmm_tmp, xmm_tmp);
                            cmpss(xmm_tmp, table_val(2), 1);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            je(init_with_zeros, T_NEAR);

                            movss(xmm_tmp, xmm_iw_im);
                            cvtps2dq(xmm_tmp, xmm_tmp);
//                            paddd(xmm_tmp, table_val(5));
                            cmpss(xmm_tmp, table_val(6), 0x0e);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(init_with_zeros, T_NEAR);
                        } else {
                            movss(xmm_tmp, xmm_iw_im);
                            cmpss(xmm_tmp, table_val(0), 1);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(init_with_zeros, T_NEAR);

                            cmpss(xmm_iw_im, table_val(2), 1);
                            movq(reg_tmp_64, xmm_iw_im);
                            cmp(reg_tmp_32, 0);
                            je(init_with_zeros, T_NEAR);
                        }

                        // interpolation calculation

                        movd(xmm_cur_height, table_val(3));
                        psubd(xmm_cur_height, xmm_ih_in);

                        roundps(xmm_h_low, xmm_map_h, 1);
                        cvtps2dq(xmm_h_low, xmm_h_low);
                        maxss(xmm_h_low, table_val(0));

                        if (jcp_.with_bi_pad) {
                            movdqu(xmm_h_high, xmm_h_low);
                            paddd(xmm_h_high, table_val(5));
                        } else {
                            roundps(xmm_h_high, xmm_map_h, 2);
                            cvtps2dq(xmm_h_high, xmm_h_high);
                            minss(xmm_h_high, xmm_cur_height);
                        }

                        movd(xmm_cur_width, table_val(4));
                        psubd(xmm_cur_width, xmm_iw_in);

                        roundps(xmm_w_low, xmm_map_w, 1);
                        cvtps2dq(xmm_w_low, xmm_w_low);
                        maxss(xmm_w_low, table_val(0));

                        if (jcp_.with_bi_pad) {
                            movdqu(xmm_w_high, xmm_w_low);
                            paddd(xmm_w_high, table_val(5));
                        } else {
                            roundps(xmm_w_high, xmm_map_w, 2);
                            cvtps2dq(xmm_w_high, xmm_w_high);
                            minss(xmm_w_high, xmm_cur_width);
                        }

                        cvtdq2ps(xmm_tmp, xmm_w_low);
                        subss(xmm_lw, xmm_tmp);

                        movss(xmm_hw, table_val(5));
                        cvtdq2ps(xmm_hw, xmm_hw);
                        subss(xmm_hw, xmm_lw);

                        cvtdq2ps(xmm_tmp, xmm_h_low);
                        subss(xmm_lh, xmm_tmp);

                        movss(xmm_hh, table_val(5));
                        cvtdq2ps(xmm_hh, xmm_hh);
                        subss(xmm_hh, xmm_lh);

                        movss(xmm_w1, xmm_hh);
                        mulss(xmm_w1, xmm_hw);
                        uni_vbroadcastss(vmm_w1, xmm_w1);

                        movss(xmm_w2, xmm_hh);
                        mulss(xmm_w2, xmm_lw);
                        uni_vbroadcastss(vmm_w2, xmm_w2);

                        movss(xmm_w3, xmm_lh);
                        mulss(xmm_w3, xmm_hw);
                        uni_vbroadcastss(vmm_w3, xmm_w3);

                        movss(xmm_w4, xmm_lh);
                        mulss(xmm_w4, xmm_lw);
                        uni_vbroadcastss(vmm_w4, xmm_w4);

                        int simd_w = vlen / jcp_.typesize_in;
                        mov(reg_ic_iter, ic_per_def_group);
                        L(ic_loop_main);
                        {
                            cmp(reg_ic_iter, simd_w);
                            jl(ic_loop_tail, T_NEAR);

                            size_t input_buffer_off = (size_t) kh * jcp_.kw * jcp_.ic + kw * jcp_.ic;

                            pmovsxdq(xmm_v1_off, xmm_v1_off);
                            movq(reg_tmp_64, xmm_v1_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            // w_low >= 0
                            movups(xmm_tmp, xmm_w_low);
                            pcmpgtd(xmm_tmp, table_val(0));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // jne(v1_condition_end_main, T_NEAR);

                            // h_low >= 0
                            movups(xmm_tmp, xmm_h_low);
                            pcmpgtd(xmm_tmp, table_val(0));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // jne(v1_condition_end_main, T_NEAR);

                            uni_vmovups(vmm_v1, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v1, vmm_v1, vmm_w1);
                            L(v1_condition_end_main);


                            pmovsxdq(xmm_v2_off, xmm_v2_off);
                            movq(reg_tmp_64, xmm_v2_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);

                            // w_high <= cur_width - 1
                            movups(xmm_tmp, xmm_w_high);
                            psubd(xmm_tmp, table_val(0));
                            pcmpgtd(xmm_tmp, table_val(4));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // je(v2_condition_end_main, T_NEAR);

                            // h_low >= 0
                            movups(xmm_tmp, xmm_h_low);
                            pcmpgtd(xmm_tmp, table_val(0));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // jne(v2_condition_end_main, T_NEAR);

                            uni_vmovups(vmm_v2, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v2, vmm_v2, vmm_w2);
                            L(v2_condition_end_main);

                            pmovsxdq(xmm_v3_off, xmm_v3_off);
                            movq(reg_tmp_64, xmm_v3_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);

                            // w_low >= 0
                            movups(xmm_tmp, xmm_w_low);
                            pcmpgtd(xmm_tmp, table_val(0));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // jne(v3_condition_end_main, T_NEAR);

                            // h_high <= cur_height
                            movups(xmm_tmp, xmm_h_high);
                            psubd(xmm_tmp, table_val(0));
                            pcmpgtd(xmm_tmp, table_val(3));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // je(v3_condition_end_main, T_NEAR);

                            uni_vmovups(vmm_v3, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v3, vmm_v3, vmm_w3);
                            L(v3_condition_end_main);

                            pmovsxdq(xmm_v4_off, xmm_v4_off);
                            movq(reg_tmp_64, xmm_v4_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);

                            // w_high <= cur_width
                            movups(xmm_tmp, xmm_w_high);
                            psubd(xmm_tmp, table_val(0));
                            pcmpgtd(xmm_tmp, table_val(3));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // je(v4_condition_end_main, T_NEAR);

                            // h_high <= cur_height
                            movups(xmm_tmp, xmm_h_high);
                            psubd(xmm_tmp, table_val(0));
                            pcmpgtd(xmm_tmp, table_val(4));
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            // je(v4_condition_end_main, T_NEAR);

                            uni_vmovups(vmm_v4, ptr[reg_tmp_64]);
                            uni_vmulps(vmm_v4, vmm_v4, vmm_w4);
                            L(v4_condition_end_main);

                            uni_vaddps(vmm_v1, vmm_v1, vmm_v2);
                            uni_vaddps(vmm_v1, vmm_v1, vmm_v3);
                            uni_vaddps(vmm_v1, vmm_v1, vmm_v4);
                            uni_vmovups(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], vmm_v1);

                            add(aux2_reg_input, simd_w * jcp_.typesize_in);
                            add(aux3_reg_input_buffer, simd_w * jcp_.typesize_in);
                            sub(reg_ic_iter, simd_w);
                            jmp(ic_loop_main, T_NEAR);
                        }

                        L(ic_loop_tail);
                        {
                            cmp(reg_ic_iter, 1);
                            jl(loop_end, T_NEAR);

                            size_t input_buffer_off = (size_t) kh * jcp_.kw * jcp_.ic + kw * jcp_.ic;

                            movss(xmm_v1, table_val(0));
                            // w_low >= 0
                            movq(reg_tmp_64, xmm_w_low);
                            cmp(reg_tmp_32, 0);
                            jl(v1_condition_end_tail, T_NEAR);

                            // h_low >= 0
                            movq(reg_tmp_64, xmm_h_low);
                            cmp(reg_tmp_32, 0);
                            jl(v1_condition_end_tail, T_NEAR);

                            movups(xmm_v1_off, table_val(2));
                            cvtps2dq(xmm_v1_off, xmm_v1_off);
                            pmulld(xmm_v1_off, xmm_h_low);
                            paddd(xmm_v1_off, xmm_w_low);
                            pmovsxdq(xmm_v1_off, xmm_v1_off);

                            movq(reg_tmp_64, xmm_v1_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v1, ptr[reg_tmp_64]);
                            mulss(xmm_v1, xmm_w1);
                            L(v1_condition_end_tail);

                            movss(xmm_v2, table_val(0));
                            // w_high <= cur_width - 1
                            movq(xmm_tmp, xmm_w_high);
                            pcmpgtd(xmm_tmp, xmm_cur_width);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(v2_condition_end_tail, T_NEAR);

                            // h_low >= 0
                            movq(reg_tmp_64, xmm_h_low);
                            cmp(reg_tmp_32, 0);
                            jl(v2_condition_end_tail, T_NEAR);


                            movups(xmm_v2_off, table_val(2));
                            cvtps2dq(xmm_v2_off, xmm_v2_off);
                            pmulld(xmm_v2_off, xmm_h_low);
                            paddd(xmm_v2_off, xmm_w_high);
                            pmovsxdq(xmm_v2_off, xmm_v2_off);

                            movq(reg_tmp_64, xmm_v2_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v2, ptr[reg_tmp_64]);
                            mulss(xmm_v2, xmm_w2);
                            L(v2_condition_end_tail);

                            movss(xmm_v3, table_val(0));
                            // w_low >= 0
                            movq(reg_tmp_64, xmm_w_low);
                            cmp(reg_tmp_32, 0);
                            jl(v3_condition_end_tail, T_NEAR);

                            // h_high <= cur_height - 1
                            movq(xmm_tmp, xmm_h_high);
                            pcmpgtd(xmm_tmp, xmm_cur_height);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(v3_condition_end_tail, T_NEAR);

                            movups(xmm_v3_off, table_val(2));
                            cvtps2dq(xmm_v3_off, xmm_v3_off);
                            pmulld(xmm_v3_off, xmm_h_high);
                            paddd(xmm_v3_off, xmm_w_low);
                            pmovsxdq(xmm_v3_off, xmm_v3_off);

                            movq(reg_tmp_64, xmm_v3_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);
                            movss(xmm_v3, ptr[reg_tmp_64]);
                            mulss(xmm_v3, xmm_w3);

                            L(v3_condition_end_tail);

                            movss(xmm_v4, table_val(0));
                            // w_high <= cur_width - 1
                            movq(xmm_tmp, xmm_w_high);
                            pcmpgtd(xmm_tmp, xmm_cur_width);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(v4_condition_end_tail, T_NEAR);

                            // h_high <= cur_height - 1
                            movq(xmm_tmp, xmm_h_high);
                            pcmpgtd(xmm_tmp, xmm_cur_height);
                            movq(reg_tmp_64, xmm_tmp);
                            cmp(reg_tmp_32, 0);
                            jne(v4_condition_end_tail, T_NEAR);


                            movups(xmm_v4_off, table_val(2));
                            cvtps2dq(xmm_v4_off, xmm_v4_off);
                            pmulld(xmm_v4_off, xmm_h_high);
                            paddd(xmm_v4_off, xmm_w_high);
                            pmovsxdq(xmm_v4_off, xmm_v4_off);

                            movq(reg_tmp_64, xmm_v4_off);
                            imul(reg_tmp_64, reg_tmp_64, jcp_.ic * jcp_.typesize_in);
                            add(reg_tmp_64, aux2_reg_input);

                            movss(xmm_v4, ptr[reg_tmp_64]);
                            mulss(xmm_v4, xmm_w4);
                            L(v4_condition_end_tail);

                            addss(xmm_v1, xmm_v2);
                            addss(xmm_v1, xmm_v3);
                            addss(xmm_v1, xmm_v4);

                            if (jcp_.with_modulation) {
                                size_t modulation_offset = ((kh * jcp_.kw + kw) * jcp_.oh * jcp_.ow) + ow;
                                mulss(xmm_v1, ptr[reg_modulation + modulation_offset * jcp_.typesize_modulation]);
                            }

                            movss(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], xmm_v1);

                            add(aux2_reg_input, jcp_.typesize_in);
                            add(aux3_reg_input_buffer, jcp_.typesize_in);
                            sub(reg_ic_iter, 1);
                            jmp(ic_loop_tail, T_NEAR);
                        }

                        jmp(loop_end, T_NEAR);

                        L(init_with_zeros);

                        mov(reg_ic_iter, 0);
                        L(ic_loop_zeros);
                        {
                            cmp(reg_ic_iter, ic_per_def_group);
                            je(loop_end, T_NEAR);

                            size_t input_buffer_off = (size_t) kh * jcp_.kw * jcp_.ic + kw * jcp_.ic;

                            pxor(xmm_tmp, xmm_tmp);
                            movss(ptr[aux3_reg_input_buffer + input_buffer_off * jcp_.typesize_in], xmm_tmp);
                            add(aux3_reg_input_buffer, jcp_.typesize_in);
                            inc(reg_ic_iter);
                            jmp(ic_loop_zeros, T_NEAR);
                        }

                        L(loop_end);
                    }
                }
            }

            add(aux_reg_def_off, 2 * jcp_.kh * jcp_.kw * jcp_.oh * jcp_.ow * jcp_.typesize_off);
            if (jcp_.with_modulation) {
                add(reg_modulation, jcp_.kh * jcp_.kw * jcp_.oh * jcp_.ow * jcp_.typesize_modulation);
            }
            add(aux_reg_input, ic_per_def_group * jcp_.typesize_in);
            add(aux2_reg_input_buffer, ic_per_def_group * jcp_.typesize_in);
            inc(reg_dg_iter);
            jmp(dg_loop, T_NEAR);
        }
        L(dg_loop_end);
        if (jcp_.with_modulation) {
            pop(reg_modulation);
        }
    }

    void store_output(int ow_step, int oc_blocks_step, int oc_step) {
        int repeats = isa == cpu::x64::sse41 && oc_step > (jcp_.oc_block / 2) ? 2 : 1;

        if (jcp_.with_bias) {
            for (int r = 0; r < repeats; r++) {
                for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                    size_t bias_off = (size_t) ocb * jcp_.oc_block + r * jcp_.oc_block / 2;
                    uni_vmovups(Vmm(0), ptr[aux_reg_bias + bias_off * jcp_.typesize_bia]);

                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);

                        uni_vaddps(vmm_acc, vmm_acc, Vmm(0));
                    }
                }
            }
        }

        if (isa == avx512_common && oc_step != jcp_.oc_block) {
            int mask = (1 << oc_step) - 1;
            mov(reg_tmp_32, mask);
            kmovw(ktail_mask, reg_tmp_32);
        }

        for (int r = 0; r < repeats; r++) {
            int tail_size = isa == cpu::x64::sse41 ? std::min(jcp_.oc_block / 2, oc_step - r * jcp_.oc_block / 2) : oc_step;
            bool is_scalar_store = isa == cpu::x64::sse41 ? tail_size < jcp_.oc_block / 2 : tail_size < jcp_.oc_block;
            if (is_scalar_store) {
                for (int ow = 0; ow < ow_step; ow++) {
                    Vmm vmm_dst = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ow);
                    Xmm xmm_dst = get_xmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ow);

                    if (isa == avx512_common) {
                        size_t out_off = (size_t) ow * jcp_.oc;

                        uni_vmovups(ptr[aux_reg_output + out_off * jcp_.typesize_out], vmm_dst | ktail_mask);
                    } else {
                        for (int oc = 0; oc < tail_size; oc++) {
                            size_t out_off = (size_t) ow * jcp_.oc + oc + r * (jcp_.oc_block / 2);

                            movq(reg_tmp_64, xmm_dst);
                            mov(ptr[aux_reg_output + out_off * jcp_.typesize_out], reg_tmp_32);

                            if (isa == cpu::x64::sse41) {
                                psrldq(vmm_dst, jcp_.typesize_out);
                            } else {
                                Ymm ymm_dst = get_ymm_acc(ow);
                                Vmm vmm_tmp = Vmm(0);
                                Ymm ymm_tmp = Ymm(0);

                                vperm2i128(ymm_tmp, ymm_dst, ymm_dst, 0x01);
                                vpalignr(ymm_dst, vmm_tmp, ymm_dst, jcp_.typesize_out);
                            }
                        }
                    }
                }
            } else {
                for (int ocb = 0; ocb < oc_blocks_step; ocb++) {
                    for (int ow = 0; ow < ow_step; ow++) {
                        Vmm vmm_acc = get_vmm_acc(r * jcp_.ur_w * jcp_.nb_oc_blocking + ocb * ow_step + ow);
                        size_t out_off = (size_t) ow * jcp_.oc + ocb * jcp_.oc_block + r * (jcp_.oc_block / 2);

                        uni_vmovups(ptr[aux_reg_output + out_off * jcp_.typesize_out], vmm_acc);
                    }
                }
            }
        }
    }

    void oc_loop(int ow_step) {
        Label oc_unrolled_loop;
        Label oc_main_loop;
        Label oc_tail;

        mov(aux_reg_input_buffer, reg_input_buffer);

        push(reg_output);
        if (jcp_.with_bias)
            push(reg_bias);
        push(reg_input);
        push(reg_kernel);
        push(reg_input_buffer);

        interpolate_input(ow_step);

        pop(reg_input_buffer);
        pop(reg_kernel);
        pop(reg_input);
        if (jcp_.with_bias)
            pop(reg_bias);
        pop(reg_output);

        push(reg_ow_pos);

        mov(aux_reg_kernel, reg_kernel);
        mov(aux_reg_output, reg_output);
        if (jcp_.with_bias)
            mov(aux_reg_bias, reg_bias);

        mov(reg_oc_work, jcp_.oc);

        L(oc_unrolled_loop); {
            cmp(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);
            jl(oc_main_loop, T_NEAR);

            ic_loop(ow_step, jcp_.nb_oc_blocking, jcp_.oc_block);
            store_output(ow_step, jcp_.nb_oc_blocking, jcp_.oc_block);

            add(aux_reg_kernel, jcp_.nb_oc_blocking * jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            add(aux_reg_output, jcp_.nb_oc_blocking * jcp_.oc_block * jcp_.typesize_out);
            if (jcp_.with_bias)
                add(aux_reg_bias, jcp_.nb_oc_blocking * jcp_.oc_block * jcp_.typesize_bia);
            sub(reg_oc_work, jcp_.nb_oc_blocking * jcp_.oc_block);

            jmp(oc_unrolled_loop, T_NEAR);
        }

        L(oc_main_loop); {
            cmp(reg_oc_work, jcp_.oc_block);
            jl(oc_tail, T_NEAR);

            ic_loop(ow_step, 1, jcp_.oc_block);
            store_output(ow_step, 1, jcp_.oc_block);

            add(aux_reg_kernel, jcp_.nb_ic * jcp_.kh * jcp_.kw * jcp_.ic_block * jcp_.oc_block * jcp_.typesize_in);
            add(aux_reg_output, jcp_.oc_block * jcp_.typesize_out);
            if (jcp_.with_bias)
                add(aux_reg_bias, jcp_.oc_block * jcp_.typesize_bia);
            sub(reg_oc_work, jcp_.oc_block);

            jmp(oc_main_loop, T_NEAR);
        }

        L(oc_tail); {
            if (jcp_.oc % jcp_.oc_block != 0) {
                ic_loop(ow_step, 1, jcp_.oc % jcp_.oc_block);
                store_output(ow_step, 1, jcp_.oc % jcp_.oc_block);
            }
        }

        pop(reg_ow_pos);
    }
};

bool MKLDNNDeformableConvolutionNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ngraph::op::v1::DeformableConvolution::type_info,
                ngraph::op::v8::DeformableConvolution::type_info)) {
            errorMessage = "Node is not an instance of DeformableConvolution form the operation set v1 or v8.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDeformableConvolutionNode::MKLDNNDeformableConvolutionNode(const std::shared_ptr<ngraph::Node>& op,
                                                                 const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    auto defConvNodeBase = std::dynamic_pointer_cast<ngraph::op::util::DeformableConvolutionBase>(op);

    group = defConvNodeBase->get_group();
    deformable_group = defConvNodeBase->get_deformable_group();
    auto& strides = defConvNodeBase->get_strides();
    for (int i = 0; i < strides.size(); i++) {
        stride.push_back(strides[i]);
    }

    auto& dilations = defConvNodeBase->get_dilations();
    for (int i = 1; i <= dilations.size(); i++) {
        dilation.push_back(dilations[dilations.size() - i] - 1);
    }

    paddingL = defConvNodeBase->get_pads_begin();

    if (op->get_type_info() == ngraph::op::v8::DeformableConvolution::type_info) {
        auto defConvNode = std::dynamic_pointer_cast<ngraph::op::v8::DeformableConvolution>(op);
        with_bilinear_pad = defConvNode->get_bilinear_interpolation_pad();
    } else {
        with_bilinear_pad = false;
    }
}

void MKLDNNDeformableConvolutionNode::getSupportedDescriptors() {
    std::string errorPrefix = "DeformableConvolution layer with name '" + getName() + "' ";

    if (getParentEdges().size() != 3 && getParentEdges().size() != 4)
        IE_THROW() << errorPrefix << "has incorrect number of input edges";
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges";

    if (getParentEdgeAt(0)->getDims().ndims() != 4) {
        IE_THROW() << "Deformable convolution layer. Unsupported mode. Only 4D blobs are supported as input.";
    }

    if (getParentEdgeAt(1)->getDims().ndims() != 4) {
        IE_THROW() << errorPrefix << "doesn't support 1st input with rank: " << getParentEdgeAt(1)->getDims().ndims();
    }

    if (getParentEdgeAt(2)->getDims().ndims() != 4) {
        IE_THROW() << errorPrefix << "doesn't support 2nd input with rank: " << getParentEdgeAt(2)->getDims().ndims();
    }

    if (getChildEdgeAt(0)->getDims().ndims() != 4) {
        IE_THROW() << errorPrefix << "doesn't support output with rank: " << getChildEdgeAt(0)->getDims().ndims();
    }
}

void MKLDNNDeformableConvolutionNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

//    const int simd_w = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
//    if (group != 1 && (((getParentEdgeAt(0)->getDims()[1] / group) % simd_w != 0)
//    || ((getChildEdgeAt(0)->getDims()[1] / group) % simd_w != 0))) {
//        enforceRef = true;
//    }
    enforceRef = false;

    size_t inputsNumber = getOriginalInputsNumber();
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(inputsNumber);
    config.inConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.inConfs[1].constant = false;
    config.inConfs[1].inPlace = -1;
    config.inConfs[2].constant = false;
    config.inConfs[2].inPlace = -1;
    if (inputsNumber > 3) {
        config.inConfs[3].constant = false;
        config.inConfs[3].inPlace = -1;
    }

    config.outConfs.resize(1);
    config.outConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;

    impl_desc_type impl_type;
    if (enforceRef) {
        impl_type = impl_desc_type::ref;
    } else if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    if (!enforceRef && mayiuse(cpu::x64::sse41)) {
        // optimzed implementation
        auto dataFormat = memory::format_tag::nhwc;
        auto offFormat = memory::format_tag::nchw;
        auto weiFormat = group > 1 ? mayiuse(avx512_common) ? memory::format_tag::gOIhw16i16o : memory::format_tag::gOIhw8i8o
                                   : mayiuse(avx512_common) ? memory::format_tag::OIhw16i16o : memory::format_tag::OIhw8i8o;

        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::data_type::f32, dataFormat);
        config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::data_type::f32, offFormat);
        auto& wDims = getParentEdgeAt(2)->getDims();
        if (group > 1 && wDims.ndims() != 5) {
            auto old_dims = wDims.ToSizeVector();
            auto new_dims = InferenceEngine::SizeVector({group, div_up(old_dims[0], group)});
            for (int i = 1; i < old_dims.size(); i++) {
                new_dims.push_back(old_dims[i]);
            }
            config.inConfs[2].desc = MKLDNNMemoryDesc(MKLDNNDims(new_dims), memory::data_type::f32, weiFormat);
        } else {
            config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::data_type::f32, weiFormat);
        }
        if (inputsNumber > 3) {
            config.inConfs[3].desc = MKLDNNMemoryDesc(getParentEdgeAt(3)->getDims(), memory::data_type::f32, memory::format_tag::nchw);
        }
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::data_type::f32, dataFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_type, dataFormat});
    } else {
        // reference implementation
        auto weiFormat = group > 1 ? memory::format_tag::goihw : memory::format_tag::oihw;

        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), memory::data_type::f32, memory::format_tag::nchw);
        config.inConfs[1].desc = MKLDNNMemoryDesc(getParentEdgeAt(1)->getDims(), memory::data_type::f32, memory::format_tag::nchw);
        config.inConfs[2].desc = MKLDNNMemoryDesc(getParentEdgeAt(2)->getDims(), memory::data_type::f32, memory::format_tag::oihw);
        if (inputsNumber > 3) {
            auto dims = getParentEdgeAt(3)->getDims();
            config.inConfs[3].desc = MKLDNNMemoryDesc(getParentEdgeAt(3)->getDims(), memory::data_type::f32, memory::format_tag::nchw);
        }
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), memory::data_type::f32, memory::format_tag::nchw);
        supportedPrimitiveDescriptors.push_back({config, impl_type, weiFormat});
    }
}

void MKLDNNDeformableConvolutionNode::createPrimitive() {
    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU deformable convolution with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();

    auto srcDims = config.inConfs[0].desc.getDims();
    auto weiDims = config.inConfs[2].desc.getDims();
    auto dstDims = config.outConfs[0].desc.getDims();

    jcp.dg = deformable_group;

    jcp.ngroups = group;

    jcp.mb = srcDims[0];

    jcp.oc = dstDims[1] / jcp.ngroups;
    jcp.ic = srcDims[1] / jcp.ngroups;

    jcp.ih = srcDims[2];
    jcp.iw = srcDims[3];
    jcp.oh = dstDims[2];
    jcp.ow = dstDims[3];

    bool with_groups = group > 1;
    jcp.kh = weiDims[with_groups + 2];
    jcp.kw = weiDims[with_groups + 3];

    jcp.t_pad = paddingL[0];
    jcp.l_pad = paddingL[1];

    jcp.stride_h = stride[0];
    jcp.stride_w = stride[1];

    jcp.dilate_h = dilation[0];
    jcp.dilate_w = dilation[1];

    jcp.with_bias = false;
    jcp.with_bi_pad = with_bilinear_pad;
    jcp.with_modulation = getParentEdges().size() > 3;

    const int simd_w = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
    jcp.ic_block = simd_w;
    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    jcp.oc_block = simd_w;
    jcp.oc_padded = rnd_up(jcp.oc, jcp.oc_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.typesize_in = sizeof(float);
    jcp.typesize_off = sizeof(float);
    jcp.typesize_out = sizeof(float);
    jcp.typesize_modulation = sizeof(float);

    jcp.ur_w = mayiuse(cpu::x64::avx512_common) ? 6 : 3;
    jcp.nb_oc_blocking = !mayiuse(cpu::x64::avx2) ? 2 : 4;

    jcp.nthr = dnnl_get_max_threads();

    if (enforceRef) {
        return;
    } else if (mayiuse(cpu::x64::avx512_common)) {
        def_conv_kernel.reset(new jit_uni_def_conv_kernel_f32<cpu::x64::avx512_common>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        def_conv_kernel.reset(new jit_uni_def_conv_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        def_conv_kernel.reset(new jit_uni_def_conv_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (def_conv_kernel)
        def_conv_kernel->create_ker();
}

void MKLDNNDeformableConvolutionNode::executeReference(const float* src, const float* offsets, const float* weights, float* dst,
                                                       const std::vector<size_t>& src_strides, const std::vector<size_t>& off_strides,
                                                       const std::vector<size_t>& wei_strides, const std::vector<size_t>& dst_strides,
                                                       const float* modulation, const std::vector<size_t>& modulation_strides) {
    const bool with_groups = jcp.ngroups > 1;
    const int G = jcp.ngroups;
    const int MB = jcp.mb;
    const int OH = jcp.oh;
    const int OW = jcp.ow;
    const int IH = jcp.ih;
    const int IW = jcp.iw;

    const int OC = jcp.oc;
    const int IC = jcp.ic;
    const int KH = jcp.kh;
    const int KW = jcp.kw;

    const int KSH = jcp.stride_h;
    const int KSW = jcp.stride_w;

    const int KDH = jcp.dilate_h;
    const int KDW = jcp.dilate_w;

    const int padT = jcp.t_pad;
    const int padL = jcp.l_pad;

    const int DG = jcp.dg;

    const int channel_per_deformable_group = (IC * G) / DG;

    const bool with_bi_pad = jcp.with_bi_pad;
    auto ker = [=](int g, int mb, int oc, int oh, int ow) {
        float d = 0;
        const int h_in = oh * KSH - padT;
        const int w_in = ow * KSW - padL;

        for (int ic = 0; ic < IC; ic++) {
            const float *data_im_ptr = src + mb * src_strides[0] + (g * IC + ic) * src_strides[1] + h_in * src_strides[2] + w_in * src_strides[3];
            const int deformable_group_index = ic / channel_per_deformable_group;
            const float *data_offset_ptr = offsets + mb * off_strides[0] + (deformable_group_index * 2 * KH * KW) * off_strides[1];
            const float *modulation_offset_ptr = nullptr;
            if (modulation != nullptr) {
                modulation_offset_ptr = modulation + mb * modulation_strides[0] + (deformable_group_index * KH * KW) * modulation_strides[1];
            }

            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    const size_t data_offset_h_index = 2 * (kh * KW + kw) * off_strides[1] + oh * off_strides[2] + ow * off_strides[3];
                    const size_t data_offset_w_index = (2 * (kh * KW + kw) + 1) * off_strides[1] + oh * off_strides[2] + ow * off_strides[3];
                    const float offset_h = data_offset_ptr[data_offset_h_index];
                    const float offset_w = data_offset_ptr[data_offset_w_index];
                    float map_h = kh * (KDH + 1) + offset_h; // kernel index with offset
                    float map_w = kw * (KDW + 1) + offset_w; // kernel index with offset

                    const float h_im = h_in + map_h; // absolute pixel index with offset
                    const float w_im = w_in + map_w; // absolute pixel index with offset
                    bool skip_compute;
                    if (with_bilinear_pad) {
                        skip_compute = !(static_cast<int>(w_im) > -1 &&
                                static_cast<int>(w_im) < IW &&
                                static_cast<int>(h_im) > -1 &&
                                static_cast<int>(w_im) < IH);
                    } else {
                        skip_compute = !(h_im >= 0 && w_im >= 0 && h_im < IH && w_im < IW);
                    }
                    if (!skip_compute) {
                        const int cur_height = IH - h_in;
                        const int cur_width = IW - w_in;
                        int h_low = std::max(static_cast<int>(floorf(map_h)), 0);
                        int w_low = std::max(static_cast<int>(floorf(map_w)), 0);
                        int h_high = with_bi_pad ? h_low + 1 : std::min(static_cast<int>(ceilf(map_h)), cur_height - 1);
                        int w_high = with_bi_pad ? w_low + 1 : std::min(static_cast<int>(ceilf(map_w)), cur_width - 1);

                        float lh = map_h - h_low;
                        float lw = map_w - w_low;
                        float hh = 1 - lh, hw = 1 - lw;

                        float v1 = (w_low >= 0 && h_low >= 0) ? data_im_ptr[h_low * src_strides[2] + w_low * src_strides[3]] : 0.0f;
                        float v2 = (w_high < cur_width && h_low >= 0) ? data_im_ptr[h_low * src_strides[2] + w_high * src_strides[3]] : 0.0f;
                        float v3 = (w_low >= 0 && h_high < cur_height) ? data_im_ptr[h_high * src_strides[2] + w_low * src_strides[3]] : 0.0f;
                        float v4 = (w_high < cur_width && h_high < cur_height) ? data_im_ptr[h_high * src_strides[2] + w_high * src_strides[3]] : 0.0f;
                        float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

                        float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

                        float modulation_scalar = 1.0f;

                        if (modulation_offset_ptr != nullptr) {
                            size_t modulation_index = (kh * KW + kw) * modulation_strides[1] + oh * modulation_strides[2] + ow * modulation_strides[3];
                            modulation_scalar = modulation_offset_ptr[modulation_index];
                        }

                        const float weight = with_groups ? weights[g * wei_strides[0] + oc * wei_strides[1] + ic * wei_strides[2] + kh * wei_strides[3] +
                                                             kw * wei_strides[4]]
                                                         : weights[oc * wei_strides[0] + ic * wei_strides[1] + kh * wei_strides[2] + kw * wei_strides[3]];
                        d += val * weight * modulation_scalar;
                    }
                }
            }
        }

        return d;
    };

    parallel_nd(G, MB, OC, OH, OW,
    [&](int g, int mb, int oc, int oh, int ow) {
        dst[mb * dst_strides[0] + (g * OC + oc) * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]] = ker(g, mb, oc, oh, ow);
    });
}

void MKLDNNDeformableConvolutionNode::executeOptimized(const float* src, const float* offsets, const float* weights, float* dst,
                                                       const std::vector<size_t>& src_strides, const std::vector<size_t>& off_strides,
                                                       const std::vector<size_t>& dst_strides, const float* modulation,
                                                       const std::vector<size_t>& modulation_strides) {
    size_t buffer_size = (size_t)jcp.nthr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic * jcp.typesize_in;
    std::vector<float> input_buffer(buffer_size, 0);
    float* input_buffer_ptr = &input_buffer[0];

    parallel_for3d(jcp.mb, jcp.ngroups, jcp.oh, [&](size_t n, size_t g, size_t oh) {
        auto ithr = parallel_get_thread_num();

        auto par_conv = jit_def_conv_call_args();

        const size_t _oc = g * jcp.nb_oc;
        const size_t _ic = g * jcp.nb_ic;

        par_conv.src = &src[n * src_strides[0] + _ic*jcp.ic_block * src_strides[1] +
                            (oh * jcp.stride_h - jcp.t_pad) * src_strides[2] - jcp.l_pad * src_strides[3]];
        par_conv.off = &offsets[n * off_strides[0] + oh * off_strides[2]];
        if (modulation != nullptr) {
            par_conv.modulation = &modulation[n * modulation_strides[0] + oh * modulation_strides[2]];
        } else {
            par_conv.modulation = nullptr;
        }
        par_conv.filt = weights;
        par_conv.dst = &dst[n * dst_strides[0] + _oc*jcp.oc_block * dst_strides[1] + oh * dst_strides[2]];

        par_conv.buf = input_buffer_ptr + ithr * jcp.ur_w * jcp.kh * jcp.kw * jcp.ic;

        par_conv.oh_pos = oh;

        (*def_conv_kernel)(&par_conv);
    });
}

void MKLDNNDeformableConvolutionNode::execute(mkldnn::stream strm) {
    const size_t inputsNumber = getOriginalInputsNumber();

    auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
    auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
    auto &srcMemory2 = getParentEdgeAt(2)->getMemory();
    auto &dstMemory = getChildEdgeAt(0)->getMemory();

    const auto *src = reinterpret_cast<const float *>(srcMemory0.GetPtr());
    const auto *offsets = reinterpret_cast<const float *>(srcMemory1.GetPtr());
    const auto *weights = reinterpret_cast<const float *>(srcMemory2.GetPtr());
    float* modulation = nullptr;
    if (inputsNumber > 3) {
        modulation = reinterpret_cast<float *>(getParentEdgeAt(3)->getMemory().GetPtr());
    }

    float *dst = reinterpret_cast<float *>(dstMemory.GetPtr());

    auto selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    if (!selectedPrimitiveDescriptor)
        IE_THROW() << "CPU deformable convolution with name '" << getName() << "' doesn't have primitive descriptors.";
    auto config = selectedPrimitiveDescriptor->getConfig();

    auto src_block_desc = config.inConfs[0].desc.getBlockingDesc();
    std::vector<size_t> src_strides(src_block_desc.getStrides().size());
    for (int i = 0; i < src_strides.size(); i++) {
        src_strides[src_block_desc.getOrder()[i]] = src_block_desc.getStrides()[i];
    }

    auto dst_block_desc = config.outConfs[0].desc.getBlockingDesc();
    std::vector<size_t> dst_strides(dst_block_desc.getStrides().size());
    for (int i = 0; i < dst_strides.size(); i++) {
        dst_strides[dst_block_desc.getOrder()[i]] = dst_block_desc.getStrides()[i];
    }


    auto off_strides = config.inConfs[1].desc.getBlockingDesc().getStrides();
    auto wei_strides = config.inConfs[2].desc.getBlockingDesc().getStrides();
    InferenceEngine::SizeVector modulation_strides;
    if (inputsNumber > 3) {
        modulation_strides = config.inConfs[3].desc.getBlockingDesc().getStrides();
    }


    if (def_conv_kernel) {
        executeOptimized(src, offsets, weights, dst, src_strides, off_strides, dst_strides, modulation, modulation_strides);
    } else {
        executeReference(src, offsets, weights, dst, src_strides, off_strides, wei_strides, dst_strides, modulation, modulation_strides);
    }
}

bool MKLDNNDeformableConvolutionNode::created() const {
    return getType() == DeformableConvolution;
}

InferenceEngine::Precision MKLDNNDeformableConvolutionNode::getRuntimePrecision() const {
    return MKLDNNExtensionUtils::getMaxPrecision(getInputPrecisions());
}

REG_MKLDNN_PRIM_FOR(MKLDNNDeformableConvolutionNode, DeformableConvolution);
