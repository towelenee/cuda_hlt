#define BOOST_TEST_MODULE "Test_Catboost_CPU"

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include "Catboost.h"

std::vector<std::vector<float>> read_csv_data_file(
  const std::string& data_path
);

BOOST_AUTO_TEST_CASE(Test_Catboost_CPU)
{
  const std::string model_path = "/home/popov/Documents/Practice/data/MuID-Run2-MC-570-v1.cb";
  const std::string signal_data_path = "/home/popov/Documents/Practice/data/signal.csv";
  
  if ( !boost::filesystem::exists(model_path) ){
   std::cout << "Can't find model file: " << model_path << std::endl;
  }
  CatboostEvaluator evaluator(model_path);
  int model_float_feature_num = (int)evaluator.GetFloatFeatureCount();  
  BOOST_CHECK_EQUAL(model_float_feature_num, 20);

  std::vector<std::vector<float>> features;
  float result;
  if ( !boost::filesystem::exists(signal_data_path) ){
   std::cout << "Can't find data file: " << signal_data_path << std::endl;
  } else {
    features = read_csv_data_file(signal_data_path);
  }
  for( const std::vector<float> event : features ) {
    result = evaluator.Apply(event, NCatboostStandalone::EPredictionType::Probability);
    BOOST_CHECK((0 <= result)&&(result <= 1));
  }
}

std::vector<std::vector<float>> read_csv_data_file(
  const std::string& data_path
) {
  std::vector<std::vector<float>> features;
  std::ifstream file(data_path);
  std::vector<float> event;
  std::string line;
  std::string cell;
   while( file ) {
    std::getline(file,line);
    std::stringstream lineStream(line);
    event.clear();
     while( std::getline( lineStream, cell, ',' ) )
      event.push_back( std::stof(cell) );
    if(!event.empty())
      features.push_back(event);
  }
  return features;
} 