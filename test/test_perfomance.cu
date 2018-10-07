#define BOOST_TEST_MODULE "Test_Perfomance"
#define EVENTS_NUMBER 1000000
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <random>
#include "Catboost.h"
#include "Evaluator.cuh"
#include "GenerateBinFeatures.cuh"

#define cudaCheck(stmt) {                                \
  cudaError_t err = stmt;                                \
  if (err != cudaSuccess){                               \
    std::cerr << "Failed to run " << #stmt << std::endl; \
    std::cerr << cudaGetErrorString(err) << std::endl;   \
    throw std::invalid_argument("cudaCheck failed");     \
  }                                                      \
}

BOOST_AUTO_TEST_CASE(Test_Perfomance)
{
  cudaEvent_t start;
  cudaEvent_t stop;

  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

  const std::string model_path = "/home/popov/Documents/Practice/data/MuID-Run2-MC-570-v1.cb";
  const std::string signal_data_path = "/home/popov/Documents/Practice/data/signal.csv";
  if ( !boost::filesystem::exists(model_path) ){
   std::cout << "Can't find model file: " << model_path << std::endl;
  }

  const int number_of_events = EVENTS_NUMBER;
  int model_bin_feature_num = 0;
  int *host_tree_sizes, *dev_tree_sizes;
  int *host_border_nums, *dev_border_nums;
  int **host_tree_splits, **dev_tree_splits;
  float *host_catboost_output, *dev_catboost_output;
  float **host_borders, **dev_borders;
  float **host_features, **dev_features;
  double **host_leaf_values, **dev_leaf_values;
  unsigned char *dev_bin_features; 

  CatboostEvaluator evaluator(model_path);
  int model_float_feature_num = (int)evaluator.GetFloatFeatureCount();  
  BOOST_CHECK_EQUAL(model_float_feature_num, 20);
  const NCatBoostFbs::TObliviousTrees* ObliviousTrees = evaluator.GetObliviousTrees();
  int tree_num = ObliviousTrees->TreeSizes()->size();
  const int* treeSplitsPtr_flat = ObliviousTrees->TreeSplits()->data();
  const double* leafValuesPtr_flat = ObliviousTrees->LeafValues()->data();

  std::vector<std::vector<float>> features;
  std::vector<float> event(model_float_feature_num);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> dis(-5.0, 5.0);
  for (size_t j = 0; j < model_float_feature_num; ++j) {
    event[j] = dis(mt);
  }
  for (size_t i = 0; i < number_of_events; ++i) {
    features.push_back(event);
  }

  cudaCheck(cudaMallocHost((void***)&host_features, number_of_events * sizeof(float*)));
  cudaCheck(cudaMallocHost((void***)&host_borders, model_float_feature_num * sizeof(float*)));
  cudaCheck(cudaMallocHost((void**)&host_border_nums, model_float_feature_num * sizeof(int)));
  cudaCheck(cudaMallocHost((void***)&host_leaf_values, tree_num * sizeof(double*)));
  cudaCheck(cudaMallocHost((void***)&host_tree_splits, tree_num * sizeof(int*)));
  cudaCheck(cudaMallocHost((void**)&host_catboost_output, number_of_events * sizeof(float)));
  cudaCheck(cudaMallocHost((void**)&host_tree_sizes, tree_num * sizeof(int)));

  int index = 0;
  for (const auto& ff : *ObliviousTrees->FloatFeatures()) {
    int border_num = ff->Borders()->size();
    host_border_nums[index] = border_num;
    model_bin_feature_num += border_num;
    cudaCheck(cudaMalloc((void**)&host_borders[index], border_num*sizeof(float)));
    cudaCheck(cudaMemcpy(host_borders[index], ff->Borders()+1, border_num*sizeof(float),cudaMemcpyHostToDevice));
    index++;
  }
  for (int i = 0; i < tree_num; i++) {
    host_tree_sizes[i] = ObliviousTrees->TreeSizes()->Get(i);
  }
  for (int i = 0; i < tree_num; i++) {
    int depth = host_tree_sizes[i];
    cudaCheck(cudaMalloc((void**)&host_leaf_values[i], (1 << depth)*sizeof(double)));
    cudaCheck(cudaMemcpy(host_leaf_values[i], leafValuesPtr_flat, (1 << depth)*sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc((void**)&host_tree_splits[i], depth*sizeof(int)));
    cudaCheck(cudaMemcpy(host_tree_splits[i], treeSplitsPtr_flat, depth*sizeof(int), cudaMemcpyHostToDevice));
  
    leafValuesPtr_flat += (1 << depth);
    treeSplitsPtr_flat += depth;
  }

  for (int i = 0; i < number_of_events; ++i) {
    cudaCheck(cudaMalloc((void**)&host_features[i], model_float_feature_num*sizeof(float)));
    cudaCheck(cudaMemcpy(host_features[i], features[i].data(), model_float_feature_num*sizeof(float),cudaMemcpyHostToDevice));
  }

  cudaCheck(cudaMalloc((void***)&dev_features, number_of_events * sizeof(float*)));
  cudaCheck(cudaMalloc((void***)&dev_borders, model_float_feature_num * sizeof(float*)));
  cudaCheck(cudaMalloc((void**)&dev_border_nums, model_float_feature_num * sizeof(int)));
  cudaCheck(cudaMalloc((void***)&dev_leaf_values, tree_num * sizeof(double*)));
  cudaCheck(cudaMalloc((void***)&dev_tree_splits, tree_num * sizeof(int*)));
  cudaCheck(cudaMalloc((void**)&dev_catboost_output, number_of_events * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&dev_tree_sizes, tree_num * sizeof(int)));
  cudaCheck(cudaMalloc((void**)&dev_bin_features, number_of_events * model_bin_feature_num * sizeof(char)));

  cudaCheck(cudaMemcpyAsync(dev_borders, host_borders, model_float_feature_num * sizeof(float*), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dev_features, host_features, number_of_events * sizeof(float*), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dev_border_nums, host_border_nums, model_float_feature_num * sizeof(int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dev_tree_splits, host_tree_splits, tree_num * sizeof(int*), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dev_leaf_values, host_leaf_values, tree_num * sizeof(double*), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpyAsync(dev_tree_sizes, host_tree_sizes, tree_num * sizeof(int), cudaMemcpyHostToDevice));

  cudaEventRecord(start, 0);
  gen_bin_features<<<dim3(number_of_events), dim3(model_float_feature_num)>>>(
    dev_borders,
    dev_features,
    dev_border_nums,
    dev_bin_features,
    number_of_events,
    model_bin_feature_num
  );
  
  catboost_evaluator<<<dim3(number_of_events), dim3(32), 32*sizeof(float)>>>(
    dev_tree_splits,
    dev_leaf_values,
    dev_tree_sizes,
    dev_catboost_output,
    dev_bin_features,
    tree_num,
    number_of_events,
    model_bin_feature_num
  );
  cudaEventRecord(stop, 0);

  float time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  BOOST_TEST_MESSAGE("GPU compute time: " << time);

  cudaCheck(cudaMemcpyAsync(host_catboost_output, dev_catboost_output, number_of_events*sizeof(float), cudaMemcpyDeviceToHost));
}