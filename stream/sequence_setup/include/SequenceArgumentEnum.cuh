#pragma once

namespace seq {
/**
 * seq_enum_t contains all steps of the sequence in the expected
 *            order of execution.
 */
enum seq_enum_t {
  estimate_input_size,
  prefix_sum_reduce,
  prefix_sum_single_block,
  prefix_sum_scan,
  masked_velo_clustering,
  calculate_phi_and_sort,
  fill_candidates,
  search_by_triplet,
  weak_tracks_adder,
  copy_and_prefix_sum_single_block,
  copy_velo_track_hit_number,
  prefix_sum_reduce_velo_track_hit_number,
  prefix_sum_single_block_velo_track_hit_number,
  prefix_sum_scan_velo_track_hit_number,
  consolidate_tracks,
  ut_calculate_number_of_hits,
  prefix_sum_reduce_ut_hits,
  prefix_sum_single_block_ut_hits,
  prefix_sum_scan_ut_hits,
  decode_raw_banks,
  sort_by_y,
  veloUT,
  estimate_cluster_count,
  prefix_sum_reduce_scifi_hits,
  prefix_sum_single_block_scifi_hits,
  prefix_sum_scan_scifi_hits,
  raw_bank_decoder,
  scifi_sort_by_x,
  gen_bin_features,
  catboost_evaluator
};
}

namespace arg {
/**
 * arg_enum_t Arguments for all algorithms in the sequence.
 */
enum arg_enum_t {
  dev_raw_input,
  dev_raw_input_offsets,
  dev_estimated_input_size,
  dev_module_cluster_num,
  dev_module_candidate_num,
  dev_cluster_offset,
  dev_cluster_candidates,
  dev_velo_cluster_container,
  dev_tracks,
  dev_tracks_to_follow,
  dev_hit_used,
  dev_atomics_storage,
  dev_tracklets,
  dev_weak_tracks,
  dev_h0_candidates,
  dev_h2_candidates,
  dev_rel_indices,
  dev_hit_permutation,
  dev_velo_track_hit_number,
  dev_prefix_sum_auxiliary_array_2,
  dev_velo_track_hits,
  dev_velo_states,
  dev_ut_raw_input,
  dev_ut_raw_input_offsets,
  dev_ut_hit_offsets,
  dev_ut_hit_count,
  dev_prefix_sum_auxiliary_array_3,
  dev_ut_hits,
  dev_ut_hit_permutations,
  dev_veloUT_tracks,
  dev_atomics_veloUT,
  dev_scifi_raw_input_offsets,
  dev_scifi_hit_count,
  dev_prefix_sum_auxiliary_array_4,
  dev_scifi_hit_permutations,
  dev_scifi_hits,
  dev_scifi_raw_input,
  dev_borders,
  dev_features,
  dev_border_nums,
  dev_bin_features,
  dev_tree_splits,
  dev_leaf_values,
  dev_tree_sizes,
  dev_catboost_output
};
}
