// sycl/src/core/conversation_state.cpp - SYCL port of src/core/conversation_state.cpp (two CUDA calls, by hand).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

// SYCL port (hand-ported: no CUDA header is included here, so dpct never saw this file): the device-wide sync and
// the any-direction copy as SYCL queue operations; errors arrive as exceptions instead of status codes.
bool sync(std::string& error) {
    try {
        dpct::get_current_device().queues_wait_and_throw();
        return true;
    } catch (sycl::exception const& e) {
        error = std::string("conversation snapshot synchronize: ") + e.what();
        return false;
    }
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    try {
        dpct::get_in_order_queue().memcpy(dst, src, bytes).wait();
        return true;
    } catch (sycl::exception const& e) {
        error = std::string("conversation snapshot running-state copy: ") + e.what();
        return false;
    }
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error, bool allow_stage_parts) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        // A layer split's checkpoints carry per-stage parts, and the request path needs them back to restore a
        // checkpoint across the cards (generate.cpp: c->stage_parts.size() != stages.size() is an error). They are
        // therefore kept in a parked image and tolerated here; a single-session image still must not carry them.
        if ((!allow_stage_parts && !c.stage_parts.empty()) || c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error) ||
        !copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
            !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
            !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

// The running-state payload conversation_checkpoint_save would write for this carve (gdn/ple/tails/dead/block_pos).
// The ids and image keys are deliberately NOT counted: a split's stages share the conversation's token list with the
// primary image, and each stage's SavedStage::live holds a reference to the same ids, not a second copy in RAM.
bool conversation_checkpoint_bytes(const ConversationCheckpoint& c, const SessionState& ss,
                                   const ModelGeometry& g, size_t& bytes, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    size_t n = 0;
    for (size_t k : {c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(n, k)) return fail(error, "checkpoint byte count overflow");
    (void) layers;
    if (!add(bytes, n)) return fail(error, "checkpoint byte count overflow");
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error,
                                 bool with_draft, bool allow_stage_parts) {
    bytes = 0;
    if (!view_validate(view, ss, g, error, allow_stage_parts)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    const uint64_t kv_slots = qsa + (with_draft ? 1 : 0);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {kv_slots, sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i < qsa; ++i) {
        const size_t n = conversation_kv_bytes(owned(ss, (size_t) i), g, upto, true);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    if (with_draft) {
        const size_t n = conversation_kv_bytes(draft, g, upto, false);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error,
                                         bool with_draft, bool allow_stage_parts) {
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error, with_draft, allow_stage_parts)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss) + (with_draft ? 1 : 0);
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i + 1 != layers;
        const auto& st = index ? owned(ss, i) : draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes, bool with_draft,
                                bool allow_stage_parts) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error, with_draft,
                                             allow_stage_parts) || !sync(error))
        return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = view.checkpoints;
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss);
    captured.kv.resize(layers + (with_draft ? 1 : 0));
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    // Under a split the drafter belongs to the LAST stage, so this image stops at its own layers.
    if (with_draft &&
        !conversation_kv_save(captured.kv.back(), draft, g, upto, false, error,
                              std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error,
                                    bool with_draft) {
    // A split parks each stage as its own SavedStage image; those are validated against the stage they belong to.
    // What is still NOT parkable is the old mid-prompt form: stage_parts describe one capture point inside a
    // checkpoint, not a restorable image, so an image carrying them stays refused.
    if (!image.stages.empty() && with_draft) return fail(error, "layer-split image on a single-session restore");
    if (image.stages.empty() && !image.live.stage_parts.empty())
        return fail(error, "layer-split parking requires per-stage images");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error, !image.stages.empty()) ||
        !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + (with_draft ? 1 : 0)) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return !with_draft || conversation_kv_validate(image.kv.back(), draft, g, upto, false, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error, bool with_draft) {
    if (!conversation_snapshot_validate(image, ss, g, draft, error, with_draft))
        return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    // Under a split the drafter's image belongs to the last stage, so this one stops at its own layers.
    if ((with_draft && !conversation_kv_restore(image.kv.back(), draft, g, upto, false, error)) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
// ---- layer-split parking: one stage at a time, each on its own card (README item 3)
// Under a split every layer's K/V and running state lives on exactly one GPU, so a park is one image per stage and
// the caller walks the stages inside an OnDevice scope for each - exactly what the mid-prompt per-stage checkpoint
// path already does. `draft` is non-null only for the stage that owns the drafter (the last one). `upto` is the
// conversation's token count: it bounds every K/V page, and it is the primary image's id count, not a per-stage one.
static bool stage_kv_slots(const SessionState& stage, const QsaState* draft, size_t& slots) {
    slots = owned_qsa(stage) + (draft ? 1 : 0);
    return true;
}
bool stage_bytes(size_t& bytes, const SessionState& stage, const ModelGeometry& g, const QsaState* draft,
                 int64_t upto, std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, stage, z, error)) return false;
    size_t slots = 0, meta = 0, tails = 0, dead = 0, positions = 0;
    stage_kv_slots(stage, draft, slots);
    const size_t layers = owned_qsa(stage);
    if (!product(meta, {(uint64_t) slots, sizeof(ConversationKv)}) || !product(tails, {layers, z.tail}) ||
        !product(dead, {layers, z.dead}) || !product(positions, {layers, z.block_pos}))
        return fail(error, "stage metadata byte count overflow");
    for (size_t k : {meta, tails, dead, positions, z.gdn, stage.ple_hist ? z.ple : 0})
        if (!add(bytes, k)) return fail(error, "stage byte count overflow");
    for (size_t j = 0; j < layers; ++j) {
        const size_t n = conversation_kv_bytes(owned(stage, j), g, upto, true);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    if (draft) {
        const size_t n = conversation_kv_bytes(*draft, g, upto, false);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}
bool stage_save(SavedStage& out, const std::vector<int32_t>& ids, const SessionState& stage,
                const ModelGeometry& g, const QsaState* draft, int64_t upto, std::string& error,
                StageKvReuse reuse, size_t* reused_bytes) {
    if (!sync(error)) return false;
    SavedStage captured;
    captured.layer_lo = stage.layer_lo; captured.layer_hi = stage.layer_hi;
    // The stage's checkpoint carries the conversation's token ids even though the image does not duplicate the
    // vector: conversation_checkpoint_validate bounds the pooled-indexer rows by ids.size(), and
    // conversation_checkpoint_restore refreshes the last pooled row ONLY when ids is non-empty. An empty ids here
    // would validate vacuously and restore a stage with stale pooled rows - silently wrong, not a crash.
    captured.live.ids = ids;
    if (!conversation_checkpoint_save(captured.live, stage, g, error)) return false;
    const size_t layers = owned_qsa(stage);
    const size_t slots = layers + (draft ? 1 : 0);
    // A retained prefix may only be reused if stage_capture_bytes validated it against THIS stage at
    // `unchanged`; the layer count is re-checked here because a mismatched retained image would otherwise be
    // silently resized onto the wrong stage's layers.
    if (!reuse.kv.empty() && reuse.kv.size() != slots)
        return fail(error, "retained stage K/V does not match this stage");
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    captured.kv.resize(slots);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_save(captured.kv[j], owned(stage, j), g, upto, true, error, unchanged, reused_bytes))
            return false;
    if (draft && !conversation_kv_save(captured.kv.back(), *draft, g, upto, false, error,
                                       std::max<int64_t>(0, unchanged - 1), reused_bytes))
        return false;
    out = std::move(captured);
    return true;
}
// The per-stage half of conversation_snapshot_capture_bytes: swap this stage's fresh K/V estimate for what a
// reused capture actually costs, so make_room is not charged for pages that are not copied. Same arithmetic as the
// primary's path - subtract the fresh estimate, add the retained capacity and its directory.
bool stage_capture_bytes(StageKvReuse& reuse, const SessionState& stage, const ModelGeometry& g,
                         const QsaState* draft, int64_t upto, size_t& bytes, std::string& error) {
    if (reuse.kv.empty()) return true;
    const size_t slots = owned_qsa(stage) + (draft ? 1 : 0);
    if (reuse.kv.size() != slots || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > upto)
        return fail(error, "invalid retained stage K/V prefix");
    for (size_t i = 0; i < slots; ++i) {
        const bool index = i + 1 != slots;
        const auto& st = index ? owned(stage, i) : *draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, upto, index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, upto, index, retained, error)) return false;
        if (fresh > bytes) return fail(error, "retained stage K/V accounting underflow");
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained stage K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - slots, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained stage K/V directory overflow");
    return true;
}
bool stage_validate(const SavedStage& image, const SessionState& stage, const ModelGeometry& g,
                    const QsaState* draft, int64_t upto, std::string& error) {
    if (image.layer_lo != stage.layer_lo || image.layer_hi != stage.layer_hi)
        return fail(error, "stage image from another session layer range");
    if (!conversation_checkpoint_validate(image.live, stage, g, error)) return false;
    const size_t layers = owned_qsa(stage);
    if (image.kv.size() != layers + (draft ? 1 : 0)) return fail(error, "invalid stage K/V layer count");
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(stage, j), g, upto, true, error)) return false;
    return !draft || conversation_kv_validate(image.kv.back(), *draft, g, upto, false, error);
}
ConversationRestore stage_restore(const SavedStage& image, SessionState& stage, const ModelGeometry& g,
                                  const QsaState* draft, int64_t upto, std::string& error) {
    if (!stage_validate(image, stage, g, draft, upto, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const size_t layers = owned_qsa(stage);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_restore(image.kv[j], owned(stage, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (draft && !conversation_kv_restore(image.kv.back(), *draft, g, upto, false, error))
        return ConversationRestore::transfer_failed;
    // The recurrent state last: a failed K/V transfer must not leave half-restored running state behind.
    if (!conversation_checkpoint_restore(image.live, stage, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
