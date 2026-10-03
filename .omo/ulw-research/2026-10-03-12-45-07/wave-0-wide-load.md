# Wave 0 — P0-4 wide-load return (complete, st_01a101cf)

## Three mechanisms (distinct from L1)
1. Q8_0/Q8_1 wide MMVQ + DMMV ESIMD (#29186): reorder_vec_dot_q8_0_wide (vecdotq.hpp:419-446) uses single sycl::int4 reinterpret_cast (4 dwords) + 4 dp4a; gated by GGML_SYCL_MMVQ_WIDE (default 1). DMMV ESIMD q8_0_mac_stripe<NBLK> (dmmv.cpp:2012-2041) + dequantize_mul_mat_vec_q8_0_reorder_esimd (2046-2098): block_load<int8_t,32*NBLK>, WG scales with nblk_row (2106-2117).
2. Q2_K reorder ESIMD (#27490): reorder_vec_dot_q_sycl<Q2_K> (vecdotq.hpp:481-511), SOA [qs nb*(QK_K/4)][scales nb*(QK_K/16)][dm nb*half2]; esimd_reorder_q_traits<Q2_K> (esimd.hpp:74-157) mac_pair block_load<uint8_t,64> qs +16 scales, 8 chunks of 32, deq=dl*q-ml.
3. Q5_K ESIMD (#26376): reorder_vec_dot_q_sycl<Q5_K> (vecdotq.hpp:673-730); esimd_reorder_q_traits<Q5_K> (esimd.hpp:374-485) with extract_bit_to_pos4 (395-406), shares unpack_scale_min_k4.

## Distinction vs L1
- L1 = the framework (mul_mat_vec_q_reorder template, generic dequantize_mul_mat_vec_reorder_esimd<T>, esimd_reorder_q_traits specialization point, SOA infra, dispatch).
- P0-4 = the specific NEW kernels: reorder_vec_dot_q8_0_wide (NO generic — new struct, only wide type), q8_0 own ESIMD kernel (QK8_0=32 not QK_K=256), esimd_reorder_q_traits<Q2_K>, <Q5_K>.
- ESIMD list (4139-4151): Q2_K..Q6_K+Q8_0; no i-quants, no Q1_0/Q4_0/Q5_0/Q5_1.

## Port map to Strata
- Strata ALREADY has: Wide32Q8(1710), Wide32IQ4NL(1730), WideQ4K(1569), WideQ5K(1612), WideIQ4XS(1628), Q6_K wide(load16_a2), Q3_K generic MMVQ, s_gemv Q8_0 act — all in native_mmvq.dp.cpp.
- GAP: Q2_K has NO decode kernel at all (only Q2_0). => highest-value P0-4 item.
- ESIMD DMMV infra absent in Strata (zero ESIMD code) — Intel-only (__INTEL_LLVM_COMPILER + <sycl/ext/intel/esimd.hpp>); MMVQ wide already covers decode.
- Q8_0 int4 wide vs load16_a2: benchmark swap worth testing.

## Acceptance
- Q2_K greenfield (match llama.cpp reorder MMVQ throughput); don't regress Q8_0/IQ4_NL (load16_a2 230-400+); IQ4_XS LUT-bound don't touch; Q4_K/Q5_K match 230-280; Q6_K match 400+.
## EXPAND
- LEAD: Q2_K is only missing dense type — port reorder MMVQ first.
- DEAD END: IQ4_XS load16_a2 2-9% slower (LUT-bound); don't pursue aligned-load for i-quants.
