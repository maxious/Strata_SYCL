# dsv41 SYCL: the DPCT1093 queue-binding audit (todo 4)

Every DPCT1093 note in `sycl/src/dsv41` sits at a `dpct::select_device(...)` call. The note
says the selected device may not be the one the code intends. This audit checks, per call-site
group, that the submissions that follow the selection are bound to the queue they must be on,
and that nothing that must be ordered sits on the wrong queue.

Count as grepped: **107** DPCT1093 notes (`ds4_model.dp.cpp` 103, `ds4_vision.dp.cpp` 4).
Groups audited: **53**. Sites judged wrong: **0**. Code changes: **none**.

## What binds what

- `Dev.st` is a device's main compute queue (`ds4_model.dp.cpp:2076`). `Chunk.cs` (`:3293`) and
  the expert-cache queues (`:3092`) are the per-device copy queues. `dpct::device_ext::create_queue()`
  returns an in-order queue here (`sycl/include/dpct/device.hpp:616-628`), and
  `dpct::get_in_order_queue()` is the per-device default in-order queue.
- A submission binds to the device that was current when its queue was created. Every
  `select_device` audited below selects the device that owns the data the following code touches,
  and every submission then uses that device's own queue handle (`d.st`, `c.cs`, `B.st[dev]`,
  `e.st`, ...), so no submission crosses to another card's queue.
- The payload-then-flag hand-back is ordered in three places, all correct:
  `xfer_put_kernel` copies the payload and sets the flag inside one kernel
  (`ds4_kernels.dp.cpp:2221-2240`); `route_publish` submits `route_publish_kernel` then
  `publish_kernel` on the same queue (`ds4_kernels.dp.cpp:2903` then `:2922`); `serve_lane`
  orders the answer payload before the seq publish with a release fence
  (`ds4_model.dp.cpp:7661-7662`) and the device waits on the same compute queue
  (`wait_seq(&resps[slot].seq, sq, s)` at `ds4_model.dp.cpp:10940`).
- Note on the brief's "second in-order queue q2": at this base (18a27e02) there is no `q2`.
  The decode answer hand-back is still the host-mapped `resps` ring, and the hand-off copies run
  on the main compute queue. The second-queue hand-back is todo 5's work. This audit checked
  the queues that exist.

## Table A - per call-site groups (site counts sum to the grepped 107)

| group | file:line range | sites | queue used | must-be queue | verdict |
| --- | --- | --- | --- | --- | --- |
| G01 halt_end | ds4_model.dp.cpp:870-870 | 1 | none (host waits on c.cs and devs[di].st) | n/a - sync only | OK |
| G02 prof_mark | ds4_model.dp.cpp:1228-1242 | 2 | d.st (event via sync_barrier) | device d's compute queue | OK |
| G03 prof_collect | ds4_model.dp.cpp:1257-1257 | 1 | none (event queries after select m.d->id) | n/a - query only | OK |
| G04 dump_layer | ds4_model.dp.cpp:1281-1281 | 1 | d.st memcpy + wait | device d's compute queue | OK |
| G05 ~Impl | ds4_model.dp.cpp:1308-1333 | 2 | destroy d.st; free via default in-order queue | device d | OK |
| G06 dalloc | ds4_model.dp.cpp:1367-1367 | 1 | default in-order queue (alloc) + d.st memset + wait | device d; zeroing ordered before use | OK |
| G07 upload | ds4_model.dp.cpp:1467-1467 | 1 | default in-order queue memcpy + wait | any queue on device d (blocking upload) | OK |
| G08 mat_gg | ds4_model.dp.cpp:1517-1517 | 1 | default in-order queue memcpy + wait | any queue on device d (blocking upload) | OK |
| G09 mat_bf16 | ds4_model.dp.cpp:1582-1582 | 1 | default in-order queue memcpy + wait | any queue on device d (blocking upload) | OK |
| G10 mat | ds4_model.dp.cpp:1668-1668 | 1 | default in-order queue memcpy + wait | any queue on device d (blocking upload) | OK |
| G11 f32 | ds4_model.dp.cpp:1744-1744 | 1 | default in-order queue memcpy + wait | any queue on device d (blocking upload) | OK |
| G12 f16raw | ds4_model.dp.cpp:1778-1778 | 1 | default in-order queue memcpy + wait | any queue on device d (blocking upload) | OK |
| G13 load | ds4_model.dp.cpp:2068-3229 | 12 | creates d.st (:2076), cache cs (:3092), chunk c.cs (:3293) after select; uploads on default in-order queue + wait | queues bound to the owning device | OK |
| G14 load_dense | ds4_model.dp.cpp:4167-4167 | 1 | device d after select; dense upload | device d | OK |
| G15 place_experts | ds4_model.dp.cpp:4424-4613 | 3 | memory query + default in-order queue memcpy + wait | the member's device | OK |
| G16 load_experts_mx | ds4_model.dp.cpp:4675-5051 | 2 | per-job select + default in-order queue memcpy + wait | the job's member device | OK |
| G17 describe | ds4_model.dp.cpp:5119-5119 | 1 | memory query only | n/a - query only | OK |
| G18 init_store | ds4_model.dp.cpp:5268-5341 | 3 | dev0 = devs[0].id; queries and the map_chunk lambda re-select dev0 | device 0 (store arena) | OK |
| G19 gpu_mapped | ds4_model.dp.cpp:5376-5376 | 1 | default in-order queue malloc (devs[0]) | device 0 | OK |
| G20 store_sync | ds4_model.dp.cpp:5494-5494 | 1 | devs[0].st memcpy | devs[0].st | OK |
| G21 reset | ds4_model.dp.cpp:5565-5575 | 2 | d.st memset + d.st wait | device d's compute queue | OK |
| G22 checkpoint | ds4_model.dp.cpp:5594-5628 | 2 | d.st memcpy + d.st wait | device d's compute queue | OK |
| G23 restore | ds4_model.dp.cpp:5684-5717 | 2 | d.st memcpy + d.st wait | device d's compute queue | OK |
| G24 slot_save | ds4_model.dp.cpp:5786-5861 | 3 | d.st wait, then default in-order queue memcpy + wait | device d | OK |
| G25 slot_load | ds4_model.dp.cpp:5966-6075 | 3 | d.st wait, then default in-order queue memcpy + wait | device d | OK |
| G26 load_vision | ds4_model.dp.cpp:6284-6284 | 1 | memory query | n/a - query only | OK |
| G27 cache_loop | ds4_model.dp.cpp:7001-7086 | 3 | cache copy queue B.st[dev] memcpy + wait; one default in-order queue readback + wait | the device's cache queue | OK |
| G28 swap_in_mx | ds4_model.dp.cpp:7135-7184 | 3 | copy queue copy_st[dev] memcpy + wait; default in-order queue + wait | the device's copy queue | OK |
| G29 cache_tick_swaps | ds4_model.dp.cpp:7324-7353 | 2 | devs[0].st / dm.st memcpy (stream order kept) | the owning device's compute queue | OK |
| G30 seed_drafter | ds4_model.dp.cpp:7719-7734 | 3 | e.st wait; d.st memcpy | each device's own queue | OK |
| G31 draft_ | ds4_model.dp.cpp:7828-8075 | 10 | d.st / e.st / hd.st; cross-device copies ordered by ext_oneapi_submit_barrier | each device's own queue | OK |
| G32 sample_row | ds4_model.dp.cpp:8096-8096 | 1 | d.st (sample_tokens + memcpy) | device d's compute queue | OK |
| G33 verify | ds4_model.dp.cpp:8145-8145 | 1 | hd.st | head device's queue | OK |
| G34 rollback | ds4_model.dp.cpp:8247-8273 | 3 | ring_restore on devs[l.dev].st / devs[di].st | the owning device's queue | OK |
| G35 layer_chunk | ds4_model.dp.cpp:8318-8318 | 1 | s = d.st | device d's compute queue | OK |
| G36 run_dev_layers | ds4_model.dp.cpp:8905-8905 | 1 | s = d.st | device d's compute queue | OK |
| G37 run_dev_chunks | ds4_model.dp.cpp:9072-9072 | 1 | s = d.st; waits on s and c.cs | d.st for compute, c.cs for copies | OK |
| G38 calibrate_paths | ds4_model.dp.cpp:9318-9318 | 1 | c.cs memcpy + wait | the chunk's copy queue | OK |
| G39 lm_end | ds4_model.dp.cpp:9683-9683 | 1 | waits on d.st and c.cs, then frees | n/a - sync only | OK |
| G40 forward_chunks | ds4_model.dp.cpp:9795-10057 | 7 | d.st / dv.st waits; dv.st memcpy; EkMark queries use the captured device id (:4112) | the owning device's queue | OK |
| G41 head_share | ds4_model.dp.cpp:10123-10123 | 1 | d.st: wait_flags then the Xfer payload-then-flag hand-back | d.st (wait and hand-back on one queue) | OK |
| G42 head_rows_dev | ds4_model.dp.cpp:10176-10176 | 1 | d.st (hc_pre_norm) | device d's compute queue | OK |
| G43 tap | ds4_model.dp.cpp:10195-10195 | 1 | d.st, barrier, async memcpy on d.st | device d's compute queue | OK |
| G44 forward | ds4_model.dp.cpp:10266-10266 | 1 | d.st wait | n/a - sync only | OK |
| G45 forward_serial | ds4_model.dp.cpp:10293-10399 | 4 | a.st / b.st / dd.st; cross-device bounce ordered by ext_oneapi_submit_barrier | each device's own queue | OK |
| G46 run_dev | ds4_model.dp.cpp:10469-10469 | 1 | s = d.st; xfer_get / xfer_put payload-then-flag hand-off on s | d.st; peers produce/consume on their own queues | OK |
| G47 replicate | ds4_model.dp.cpp:10564-10578 | 2 | o2.st barrier with the peer event, then copy; d.st | the owning device's queue | OK |
| G48 layer | ds4_model.dp.cpp:10597-10597 | 1 | s = d.st | device d's compute queue | OK |
| G49 sample | ds4_model.dp.cpp:11160-11160 | 1 | d.st | device d's compute queue | OK |
| G50 logits | ds4_model.dp.cpp:11193-11193 | 1 | default in-order queue memcpy + wait | device d | OK |
| G51 VisionTower::release | ds4_vision.dp.cpp:307-323 | 2 | destroy st_/cs_ on at_'s device; free via default in-order queue | the tower's device | OK |
| G52 VisionTower::load | ds4_vision.dp.cpp:353-353 | 1 | creates st_/cs_ after select_device(device) | the tower's device | OK |
| G53 VisionTower::encode | ds4_vision.dp.cpp:541-541 | 1 | q_of(st_) / q_of(cs_) with barriers | the tower's queues | OK |

## Table B - every queue-creation site: owning thread and the device it binds

| site | created on | device selected at creation | the queue serves | verdict |
| --- | --- | --- | --- | --- |
| `ds4_model.dp.cpp:2076` (`d.st`) | the load thread in `load()`, right after `select_device(d.id)` (`:2071`) | `d.id` = `ids[i]` | Dev `d`'s compute queue; submitted by device i's worker thread (threads mode: one thread per GPU) | MATCH |
| `ds4_model.dp.cpp:3092` (cache `cs`) | the load thread, right after `select_device(d.id)` (`:3086`) | `d.id` | the expert-cache swap queue `B.st[dev]`; submitted by the `cache_loop` thread | MATCH |
| `ds4_model.dp.cpp:3293` (`c.cs`) | the load thread; the current device is the chunk's `d.id` (`select_device` at `:3232`, kept by each `dalloc(d, ...)` at `:1370`) | the chunk's `d.id` | Chunk `c`'s copy queue; submitted by device di's worker thread | MATCH |
| `ds4_vision.dp.cpp:492` (`st_`) | the load thread in `VisionTower::load`, after `select_device(device)` (`:356`) | `device` | the tower's compute queue; used by `encode` | MATCH |
| `ds4_vision.dp.cpp:497` (`cs_`) | the load thread in `VisionTower::load` | `device` | the tower's copy queue; used by `encode` | MATCH |

All five match. SYCL queues are not thread-affine: the binding comes from the device that is
current on the creating thread, and every creating thread selected the owning device first
(static reading of this tree).

## The two load-bearing queue-bound sites

The answer-payload hand-back (host side, `ds4_model.dp.cpp:7648-7662`):

```cpp
                        float* o = rp.part[t];
...
            std::atomic_thread_fence(std::memory_order_release);
            *(volatile unsigned*) &rp.seq = next;   // the answer: the release fence above orders the payload before it
```

The routing-flag publish (`ds4_kernels.dp.cpp:2160-2176`, submitted on the compute queue at
`ds4_kernels.dp.cpp:2922` from `route_publish`, called at `ds4_model.dp.cpp:10915`):

```cpp
void publish_kernel(Req* req, int T, unsigned seq, int na0, int na1) {
...
    strata::sys_store((volatile unsigned*) &req->seq, seq);
```

Both publish the flag after the payload. The first with a release fence before the host store;
the second by two submissions on one in-order queue. Neither is left on a queue that could
reorder the flag ahead of the payload.

## Verdict

No submission that must be ordered is on the wrong queue. No code change was needed, so this
file is the deliverable and the commit carries only this audit.

Evidence: `.omo/evidence/dsv41-sycl-dpct-and-xpu-test/task-4-dsv41-sycl-dpct-and-xpu-test.log`

## Acceptance runs: the two-card smoke and the single-card control (2026-10-26)

Both runs use `--ids 271 --n 8 --max-context 512`, `SYCL_BE=PI_LEVEL_ZERO`, with both
streams captured in `/tmp/dpct_run/`. Neither run reached decode, so there is no `prompt:`
line to quote and no throughput claim. The logs carry the failures verbatim.

| run | command | result | prompt: line | what it shows |
| --- | --- | --- | --- | --- |
| two-card smoke | `--gpus 0,1 --ids 271 --n 8 --max-context 512 > /tmp/dpct_run/run_t4_two.log 2>&1` | `RUN_TWO_RC=134` (abort; identical in 3 attempts) | none (`PROMPT_TWO_RC=1`) | dies during load at the cross-device bounce: `opencl backend failed with error: 13 (UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST)` then `maya-ds4: DPCT_CHECK_ERROR((b.st)->ext_oneapi_submit_barrier({*a.ev_out})): error` (`ds4_model.dp.cpp:10362`) |
| single-card control | `--gpus 1 --ids 271 --n 8 --max-context 512 > /tmp/dpct_run/run_t4_one.log 2>&1` | `RUN_ONE_RC=134` (abort; identical in 2 attempts) | none (`PROMPT_ONE_RC=1`) | `ds4: CUDA1 layers [0, 39] + head: weights 5.86 GB, experts 0.00 GB (0 of 15360 in VRAM), caches 0.00 GB, 0 MiB free` then `Enqueue process failed.` and the catch at `ds4_model.dp.cpp:11153` |

xpu-smi during the two-card run (`/tmp/dpct_run/xpu_busy_t4b.txt`, bounded sampling):
`XPU_BUSY_RC=1`. Card 0 climbs to about 14.5 GB while card 1 stays at 508 MiB against a
246 MiB idle baseline: the second physical card takes no layers and no memory.

Why the second card takes no layers (recorded, not changed - `--tp` behaviour and the device
grouping are outside this todo's fix scope):

- The layer split itself is correct: `tp=1` gives each device its own layers
  (`ds4_model.dp.cpp:2108`), and the run prints `CUDA0 layers [0, 3]` and
  `CUDA1 layers [4, 39] + head` - each device slot got its layers.
- The device behind slot 1 is not the second Arc card. `dpct::select_device(d.id)` numbers
  devices as: index 0 is the SYCL default device, then every remaining device of
  `sycl::device::get_devices(all)` in runtime order (`sycl/include/dpct/device.hpp:848-868`).
  `sycl-ls` on this PC lists `[opencl:cpu] AMD Ryzen 7 5700X3D` first, then two OpenCL Arc
  devices, then two Level Zero Arc devices - so `--gpus 1` binds the host CPU device, not
  physical card 1.
- The runs say the same thing three ways: the kernel JIT reports
  `Build program log for 'AMD Ryzen 7 5700X3D 8-Core Processor': Compilation failed,
  Unrecognized build options: -ze-opt-greater-than-4GB-buffer-required`; the device reports
  `121 GiB of graphics memory` (host RAM) and `0 MiB free`
  (`ext_intel_free_memory is not supported`); and xpu-smi shows card 1 idle throughout.
- The same mapping explains the two-card abort: the bounce hand-off barrier
  (`ds4_model.dp.cpp:10362`) carries a GPU queue's event into the CPU device's queue, which
  the backend rejects with `UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST`.

So the DPCT1093 verdict stands for the pairs as written (select, then create on the selected
device), but the dynamic control fails: the `--gpus` ordinal to device mapping does not bind
the card the user names. Mapping the ordinals onto real GPU devices is the fix; it touches the
device-index layer, which this audit records rather than changes.
