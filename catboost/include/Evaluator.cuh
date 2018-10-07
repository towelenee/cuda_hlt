__global__ void catboost_evaluator(
    const int* const* dev_tree_splits,
    const double* const* dev_leaf_values,
    const int* dev_tree_sizes,
    float* dev_catboost_output,
    const unsigned char* dev_bin_features,
    const int dev_tree_num,
    const int dev_object_num,
    const int dev_bin_feature_num
  );
   __device__ void warpReduce(
    volatile float* sdata, 
    int tid
  );
