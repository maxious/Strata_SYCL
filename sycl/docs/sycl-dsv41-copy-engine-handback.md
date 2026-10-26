# The answer's hand-back to the card: the copy engine's copy, and what the card saw

The decode path is a handshake through host memory. The card publishes a layer's
routing into a mailbox, the host answers with the part the CPU lane computed for
the experts the card does not hold, and the card waits for that answer before it
sums the experts' rows. This file is the record of what was changed, what was
measured, and what is still open. The code is in `sycl/src/dsv41`.

## What was measured

Hardware: one Intel Arc Pro B60 (`8086:e211`), driver `xe`, backend `level_zero`,
oneAPI 2026.2, NEO 26.35.39758.10. Pack:
`DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4`. Build: the `strata-dsv41` SYCL
target (`STRATA_SYCL_DSV41=ON`, `STRATA_SYCL_DSV41_EXE=ON`) on branch
`sycl/dsv41-sycl-support` at `b18f6c40` plus this change. The run:

```sh
bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; SYCL_BE=PI_LEVEL_ZERO \
  /tmp/dpct_w5_build/strata-dsv41 --pack /home/maxious/Strata-data/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO/maya-ds4 \
  --gpus 0 --ids @/tmp/q_france.ids --n 8 --max-context 512'
```

### The host's store in the polled mailbox: OBSERVED

The default path. The card's wait reads the flag the host wrote into the polled
mailbox, with the uncached read (the read that goes to memory every time). The
wait sees the flag on its first read, every time:

```
ds4: the card's waits for the answer (device 0): 64 waits (the token path 64, the window path 0), the most reads 1, the last value read 255, 0 hit the bound
prompt: 10 tokens in 160 ms (62.69 tok/s, windows of 8192)
decode: 8 tokens in 381 ms (18.39 tok/s); CPU lane: 0 of 1680 experts; the cache moved 0 (0 in all)
```

The wait's bound is never reached and the decode is as fast as it was before this
change (381 ms for 8 tokens, against 384 ms on the base build `b18f6c40`).

### The copy engine's hand-back into the card's memory: BOUNDED-GAP

The same build with `DS4_ANSWER_COPY=1`. The host hands the payload and the flag
to the card with the copy engine (a second in-order queue, the payload first and
the flag after it), and the card's wait reads the flag in the card's own memory.
The card does not see the copy's write once the slot has been used before:

```
ds4: the card's waits for the answer (device 0): 64 waits (the token path 64, the window path 0), the most reads 1000000, the last value read 191, 63 hit the bound
decode: 8 tokens in 11530 ms (0.61 tok/s); CPU lane: 0 of 1680 experts; the cache moved 0 (0 in all)
```

The wait read 191 while it waited for the seq it names: 191 is the value the same
slot held 64 requests earlier (the ring is 64 slots), so the card read the stale
line it had in its cache and never the copy's write. 63 of the last 64 waits ran
to their 1,000,000-read bound, and the decode took 30 times as long.

What the copy engine's write does and does not reach, measured on this card:

- The copy's write reaches memory: the host reads the slot back and sees the seq
  it wrote.
- The card's read of that memory does not see the copy once the card has the
  line cached. The uncached read (the read that goes to memory every time) and
  a device-scope atomic read both keep reading the card's stale line. The probe
  behind the plan's mechanism (`/tmp/dbprobe*.cpp`, 2026-10-26) read a slot the
  card had never cached and saw the copy 10 of 10; the difference is the card's
  cache line, not the copy.
- The copy engine's write covers the whole cache line, not only the 4 bytes of
  the flag: a 4-byte copy cleared the rest of the line (the wait's record and the
  pointer next to the flag with it). The answer slot is therefore a cache line of
  its own, and the host copies the slot whole.

So the copy engine's hand-back is implemented and measured, and it does not work
on this card. It is off by default (`DS4_ANSWER_COPY=1` turns it on) because it
is slower and it loses the answer.

## What was changed

- Every wait is bounded. The window path's own wait (`win_wait_kernel` in
  `sycl/src/dsv41/vendor/exl3.dp.cpp`, `mx_wait_kernel` in
  `sycl/src/dsv41/ds4_gguf.dp.cpp`) was unbounded - a wait that never sees its
  flag wedges the card, and the xe driver resets the GT node by node when a job
  runs past 5 s. It takes the answer's bound now
  (`strata::answer_spin_max()`, `STRATA_SPIN_MAX` or 1,000,000 reads).
- The answer's wait (the token path's) writes what it saw into a host-mapped
  record: the reads it made, the last value it read, whether it ran to its
  bound, and which wait it was. The engine prints the records at the end of a
  run (`the card's waits for the answer`, `the hand-off's waits`).
- A bound below what the answer's wait needs is not credible: the CPU lane's
  cold expert read takes ~40,000 reads, so a bound under that is the operator's
  mistake, and the engine ends the run and says so instead of going on with a
  missing answer.
- The copy engine's hand-back (`DS4_ANSWER_COPY=1`): the host copies the payload,
  the per-token flags and the flag into the card's memory on a second in-order
  queue (`Dev::st2`, created next to `Dev::st` on the thread that owns the
  device), the flag after the payload, and the card's wait reads the flag in the
  card's memory. A copy on the queue that runs the waiting kernel would deadlock
  (the kernel waits for the copy, the copy waits for the kernel).
- The combine kernels read the CPU lane's part from the card's memory when the
  copy engine's hand-back is on, and from the polled mailbox otherwise. No
  `sycl::atomic_ref` is used on host USM: the host-mapped flags (the hand-off's,
  the tensor-parallel all-reduce's) keep `strata::sys_load`, the one sanctioned
  host read.

## The reads that decide the answer

The card's wait reads the flag with a read that goes to memory every time. The
plan sanctioned a device-scope atomic for a flag the copy engine wrote; the
measurement above shows that read serving the card's stale line, so the uncached
read is used for both paths (the host's store in the polled mailbox and the copy
engine's write in the card's memory). Both are the same read: a plain read of
either would be served from the card's cache.

## What is still open

- The copy engine's hand-back needs a flag the card does not have cached: a
  fresh address each time, or a way to refresh the line. Neither is in this
  change.
- The CPU lane's share is still 0 of 1680 experts in this run. That is a
  different defect (the routing ids the device publishes are plain stores the
  host reads stale) and it is not this todo's.
