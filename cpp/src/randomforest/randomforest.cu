/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <cstdio>
#include <cuml/common/logger.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "randomforest_impl.cuh"

namespace ML {

using namespace MLCommon;
using namespace std;
namespace tl = treelite;

/**
 * @brief Set RF_metrics.
 * @param[in] rf_type: Random Forest type: classification or regression
 * @param[in] cfg_accuracy: accuracy.
 * @param[in] mean_abs_error: mean absolute error.
 * @param[in] mean_squared_error: mean squared error.
 * @param[in] median_abs_error: median absolute error.
 * @return RF_metrics struct with classification or regression score.
 */
RF_metrics set_all_rf_metrics(RF_type rf_type, float accuracy,
                              double mean_abs_error, double mean_squared_error,
                              double median_abs_error) {
  RF_metrics rf_metrics;
  rf_metrics.rf_type = rf_type;
  rf_metrics.accuracy = accuracy;
  rf_metrics.mean_abs_error = mean_abs_error;
  rf_metrics.mean_squared_error = mean_squared_error;
  rf_metrics.median_abs_error = median_abs_error;
  return rf_metrics;
}

/**
 * @brief Set RF_metrics for classification.
 * @param[in] cfg_accuracy: accuracy.
 * @return RF_metrics struct with classification score.
 */
RF_metrics set_rf_metrics_classification(float accuracy) {
  return set_all_rf_metrics(RF_type::CLASSIFICATION, accuracy, -1.0, -1.0,
                            -1.0);
}

/**
 * @brief Set RF_metrics for regression.
 * @param[in] mean_abs_error: mean absolute error.
 * @param[in] mean_squared_error: mean squared error.
 * @param[in] median_abs_error: median absolute error.
 * @return RF_metrics struct with regression score.
 */
RF_metrics set_rf_metrics_regression(double mean_abs_error,
                                     double mean_squared_error,
                                     double median_abs_error) {
  return set_all_rf_metrics(RF_type::REGRESSION, -1.0, mean_abs_error,
                            mean_squared_error, median_abs_error);
}

/**
 * @brief Print either accuracy metric for classification, or mean absolute error,
 *   mean squared error, and median absolute error metrics for regression.
 * @param[in] rf_metrics: random forest metrics to print.
 */
void print(const RF_metrics rf_metrics) {
  if (rf_metrics.rf_type == RF_type::CLASSIFICATION) {
    CUML_LOG_DEBUG("Accuracy: %f", rf_metrics.accuracy);
  } else if (rf_metrics.rf_type == RF_type::REGRESSION) {
    CUML_LOG_DEBUG("Mean Absolute Error: %f", rf_metrics.mean_abs_error);
    CUML_LOG_DEBUG("Mean Squared Error: %f", rf_metrics.mean_squared_error);
    CUML_LOG_DEBUG("Median Absolute Error: %f", rf_metrics.median_abs_error);
  }
}

/**
 * @brief Update labels so they are unique from 0 to n_unique_labels values.
 *   Create/update an old label to new label map per random forest.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in,out] labels_map: map of old label values to new ones.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
void preprocess_labels(int n_rows, std::vector<int>& labels,
                       std::map<int, int>& labels_map, int verbosity) {
  std::pair<std::map<int, int>::iterator, bool> ret;
  int n_unique_labels = 0;
  ML::Logger::get().setLevel(verbosity);

  CUML_LOG_DEBUG("Preprocessing labels");
  for (int i = 0; i < n_rows; i++) {
    ret = labels_map.insert(std::pair<int, int>(labels[i], n_unique_labels));
    if (ret.second) {
      n_unique_labels += 1;
    }
    auto prev = labels[i];
    labels[i] = ret.first->second;  //Update labels **IN-PLACE**
    CUML_LOG_DEBUG("Mapping %d to %d", prev, labels[i]);
  }
  CUML_LOG_DEBUG("Finished preprocessing labels");
}

/**
 * @brief Revert label preprocessing effect, if needed.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in] labels_map: map of old to new label values used during preprocessing.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
void postprocess_labels(int n_rows, std::vector<int>& labels,
                        std::map<int, int>& labels_map, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  CUML_LOG_DEBUG("Postrocessing labels");
  std::map<int, int>::iterator it;
  int n_unique_cnt = labels_map.size();
  std::vector<int> reverse_map;
  reverse_map.resize(n_unique_cnt);
  for (auto it = labels_map.begin(); it != labels_map.end(); it++) {
    reverse_map[it->second] = it->first;
  }

  for (int i = 0; i < n_rows; i++) {
    auto prev = labels[i];
    labels[i] = reverse_map[prev];
    CUML_LOG_DEBUG("Mapping %d back to %d", prev, labels[i]);
  }
  CUML_LOG_DEBUG("Finished postrocessing labels");
}

/**
 * @brief Set RF_params parameters members; use default tree parameters.
 * @param[in,out] params: update with random forest parameters
 * @param[in] cfg_n_trees: number of trees; default 1
 * @param[in] cfg_bootstrap: bootstrapping; default true
 * @param[in] cfg_rows_sample: rows sample; default 1.0f
 * @param[in] cfg_n_streams: No of parallel CUDA for training forest
 */
void set_rf_params(RF_params& params, int cfg_n_trees, bool cfg_bootstrap,
                   float cfg_rows_sample, int cfg_seed, int cfg_n_streams) {
  params.n_trees = cfg_n_trees;
  params.bootstrap = cfg_bootstrap;
  params.rows_sample = cfg_rows_sample;
  params.seed = cfg_seed;
  params.n_streams = min(cfg_n_streams, omp_get_max_threads());
  if (params.n_streams == cfg_n_streams) {
    CUML_LOG_WARN("Warning! Max setting Max streams to max openmp threads %d",
                  omp_get_max_threads());
  }
  if (cfg_n_trees < params.n_streams) params.n_streams = cfg_n_trees;
  set_tree_params(params.tree_params);  // use default tree params
}

/**
 * @brief Set all RF_params parameters members, including tree parameters.
 * @param[in,out] params: update with random forest parameters
 * @param[in] cfg_n_trees: number of trees
 * @param[in] cfg_bootstrap: bootstrapping
 * @param[in] cfg_rows_sample: rows sample
 * @param[in] cfg_n_streams: No of parallel CUDA for training forest
 * @param[in] cfg_tree_params: tree parameters
 */
void set_all_rf_params(RF_params& params, int cfg_n_trees, bool cfg_bootstrap,
                       float cfg_rows_sample, int cfg_seed, int cfg_n_streams,
                       DecisionTree::DecisionTreeParams cfg_tree_params) {
  params.n_trees = cfg_n_trees;
  params.bootstrap = cfg_bootstrap;
  params.rows_sample = cfg_rows_sample;
  params.seed = cfg_seed;
  params.n_streams = min(cfg_n_streams, omp_get_max_threads());
  if (cfg_n_trees < params.n_streams) params.n_streams = cfg_n_trees;
  set_tree_params(params.tree_params);  // use input tree params
  params.tree_params = cfg_tree_params;
}

/**
 * @brief Check validity of all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void validity_check(const RF_params rf_params) {
  ASSERT((rf_params.n_trees > 0), "Invalid n_trees %d", rf_params.n_trees);
  ASSERT((rf_params.rows_sample > 0) && (rf_params.rows_sample <= 1.0),
         "rows_sample value %f outside permitted (0, 1] range",
         rf_params.rows_sample);
  DecisionTree::validity_check(rf_params.tree_params);
}

/**
 * @brief Print all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void print(const RF_params rf_params) {
  ML::PatternSetter _("%v");
  CUML_LOG_DEBUG("n_trees: %d", rf_params.n_trees);
  CUML_LOG_DEBUG("bootstrap: %d", rf_params.bootstrap);
  CUML_LOG_DEBUG("rows_sample: %f", rf_params.rows_sample);
  CUML_LOG_DEBUG("n_streams: %d", rf_params.n_streams);
  DecisionTree::print(rf_params.tree_params);
}

/**
 * @brief Set the trees pointer of RandomForestMetaData to nullptr.
 * @param[in, out] forest: CPU pointer to RandomForestMetaData.
 */
template <class T, class L>
void null_trees_ptr(RandomForestMetaData<T, L>*& forest) {
  forest->trees = nullptr;
}

/**
 * @brief Deletes RandomForestMetaData object
 * @param[in] forest: CPU pointer to RandomForestMetaData.
 */
template <class T, class L>
void delete_rf_metadata(RandomForestMetaData<T, L>* forest) {
  delete forest;
}

template <class T, class L>
void _print_rf(const RandomForestMetaData<T, L>* forest, bool summary) {
  ML::PatternSetter _("%v");
  if (!forest || !forest->trees) {
    CUML_LOG_INFO("Empty forest");
  } else {
    CUML_LOG_INFO("Forest has %d trees, max_depth %d, and max_leaves %d",
                  forest->rf_params.n_trees,
                  forest->rf_params.tree_params.max_depth,
                  forest->rf_params.tree_params.max_leaves);
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      CUML_LOG_INFO("Tree #%d", i);
      if (summary) {
        DecisionTree::print_tree_summary<T, L>(&(forest->trees[i]));
      } else {
        DecisionTree::print_tree<T, L>(&(forest->trees[i]));
      }
    }
  }
}

/**
 * @brief Print summary for all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
void print_rf_summary(const RandomForestMetaData<T, L>* forest) {
  _print_rf(forest, true);
}

/**
 * @brief Print detailed view of all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
void print_rf_detailed(const RandomForestMetaData<T, L>* forest) {
  _print_rf(forest, false);
}

template <class T, class L>
void build_treelite_forest(ModelHandle* model,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features, int task_category) {
  // Non-zero value here for random forest models.
  // The value should be set to 0 if the model is gradient boosted trees.
  int random_forest_flag = 1;
  ModelBuilderHandle model_builder;
  // num_output_group is 1 for binary classification and regression
  // num_output_group is #class for multiclass classification which is the same as task_category
  int num_output_group = task_category > 2 ? task_category : 1;
  TREELITE_CHECK(TreeliteCreateModelBuilder(
    num_features, num_output_group, random_forest_flag, &model_builder));

  if (task_category > 2) {
    // Multi-class classification
    TREELITE_CHECK(TreeliteModelBuilderSetModelParam(
      model_builder, "pred_transform", "max_index"));
  }

  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    DecisionTree::TreeMetaDataNode<T, L>* tree_ptr = &forest->trees[i];
    TreeBuilderHandle tree_builder;

    TREELITE_CHECK(TreeliteCreateTreeBuilder(&tree_builder));
    if (tree_ptr->sparsetree.size() != 0) {
      DecisionTree::build_treelite_tree<T, L>(tree_builder, tree_ptr,
                                              num_output_group);

      // The third argument -1 means append to the end of the tree list.
      TREELITE_CHECK(
        TreeliteModelBuilderInsertTree(model_builder, tree_builder, -1));
    }
  }

  TREELITE_CHECK(TreeliteModelBuilderCommitModel(model_builder, model));
  TREELITE_CHECK(TreeliteDeleteModelBuilder(model_builder));
}

/**
 * @brief Compares the trees present in concatenated treelite forest with the trees
 *   of the forests present in the different workers. If there is a difference in the two
 *   then an error statement will be thrown.
 * @param[in] tree_from_concatenated_forest: Tree info from the concatenated forest.
 * @param[in] tree_from_individual_forest: Tree info from the forest present in each worker.
 */
void compare_trees(tl::Tree& tree_from_concatenated_forest,
                   tl::Tree& tree_from_individual_forest) {
  ASSERT(tree_from_concatenated_forest.num_nodes ==
           tree_from_individual_forest.num_nodes,
         "Error! Mismatch the number of nodes present in a tree in the "
         "concatenated forest and"
         " the tree present in the individual forests");
  for (int each_node = 0; each_node < tree_from_concatenated_forest.num_nodes;
       each_node++) {
    ASSERT(tree_from_concatenated_forest.IsLeaf(each_node) ==
             tree_from_individual_forest.IsLeaf(each_node),
           "Error! mismatch in the position of a leaf between concatenated "
           "forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.LeafValue(each_node) ==
             tree_from_individual_forest.LeafValue(each_node),
           "Error! leaf value mismatch between concatenated forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.RightChild(each_node) ==
             tree_from_individual_forest.RightChild(each_node),
           "Error! mismatch in the position of the node between concatenated "
           "forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.LeftChild(each_node) ==
             tree_from_individual_forest.LeftChild(each_node),
           "Error! mismatch in the position of the node between concatenated "
           "forest and the"
           " individual forests ");
    ASSERT(
      tree_from_concatenated_forest.SplitIndex(each_node) ==
        tree_from_individual_forest.SplitIndex(each_node),
      "Error! split index value mismatch between concatenated forest and the"
      " individual forests ");
  }
}

/**
 * @brief Compares the concatenated treelite model with the information of the forest
 *   present in the different workers. If there is a difference in the two then an error
 *   statement will be thrown.
 * @param[in] concat_tree_handle: ModelHandle for the concatenated forest.
 * @param[in] treelite_handles: List containing ModelHandles for the forest present in
 *   each worker.
 */
void compare_concat_forest_to_subforests(
  ModelHandle concat_tree_handle, std::vector<ModelHandle> treelite_handles) {
  size_t concat_forest;
  size_t total_num_trees = 0;
  for (int forest_idx = 0; forest_idx < treelite_handles.size(); forest_idx++) {
    size_t num_trees_each_forest;
    TREELITE_CHECK(TreeliteQueryNumTree(treelite_handles[forest_idx],
                                        &num_trees_each_forest));
    total_num_trees = total_num_trees + num_trees_each_forest;
  }

  TREELITE_CHECK(TreeliteQueryNumTree(concat_tree_handle, &concat_forest));

  ASSERT(
    concat_forest == total_num_trees,
    "Error! the number of trees in the concatenated forest and the sum "
    "of the trees present in the forests present in each worker are not equal");

  int concat_mod_tree_num = 0;
  tl::Model& concat_model = *(tl::Model*)(concat_tree_handle);
  for (int forest_idx = 0; forest_idx < treelite_handles.size(); forest_idx++) {
    tl::Model& model = *(tl::Model*)(treelite_handles[forest_idx]);

    ASSERT(
      concat_model.num_feature == model.num_feature,
      "Error! number of features mismatch between concatenated forest and the"
      " individual forests ");
    ASSERT(concat_model.num_output_group == model.num_output_group,
           "Error! number of output group mismatch between concatenated forest "
           "and the"
           " individual forests ");
    ASSERT(concat_model.random_forest_flag == model.random_forest_flag,
           "Error! random forest flag value mismatch between concatenated "
           "forest and the"
           " individual forests ");

    for (int indiv_trees = 0; indiv_trees < model.trees.size(); indiv_trees++) {
      compare_trees(concat_model.trees[concat_mod_tree_num + indiv_trees],
                    model.trees[indiv_trees]);
    }
    concat_mod_tree_num = concat_mod_tree_num + model.trees.size();
  }
}

/**
 * @brief Concatenates the forest information present in different workers to
 *  create a single forest. This concatenated forest is stored in a new treelite model.
 *  The model created is owned by and must be freed by the user.
 * @param[in] concat_tree_handle: ModelHandle for the concatenated forest.
 * @param[in] treelite_handles: List containing ModelHandles for the forest present in
 *   each worker.
 */
ModelHandle concatenate_trees(std::vector<ModelHandle> treelite_handles) {
  tl::Model& first_model = *(tl::Model*)treelite_handles[0];
  tl::Model* concat_model = new tl::Model;
  for (int forest_idx = 0; forest_idx < treelite_handles.size(); forest_idx++) {
    tl::Model& model = *(tl::Model*)treelite_handles[forest_idx];
    for (const tl::Tree& tree : model.trees) {
      concat_model->trees.push_back(tree.Clone());
    }
  }
  concat_model->num_feature = first_model.num_feature;
  concat_model->num_output_group = first_model.num_output_group;
  concat_model->random_forest_flag = first_model.random_forest_flag;
  concat_model->param = first_model.param;
  return concat_model;
}

/**
 * @defgroup Random Forest Classification - Fit function
 * @brief Build (i.e., fit, train) random forest classifier for input data.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per
 *   training sample. Device pointer.
 *   Assumption: labels were preprocessed to map to ascending numbers from 0;
 *   needed for current gini impl. in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void fit(const cumlHandle& user_handle, RandomForestClassifierF*& forest,
         float* input, int n_rows, int n_cols, int* labels, int n_unique_labels,
         RF_params rf_params, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<float, int>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfClassifier<float>> rf_classifier =
    std::make_shared<rfClassifier<float>>(rf_params);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels,
                     n_unique_labels, forest);
}

void fit(const cumlHandle& user_handle, RandomForestClassifierD*& forest,
         double* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels, RF_params rf_params, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<double, int>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfClassifier<double>> rf_classifier =
    std::make_shared<rfClassifier<double>>(rf_params);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels,
                     n_unique_labels, forest);
}
/** @} */

/**
 * @defgroup Random Forest Classification - Predict function
 * @brief Predict target feature for input data; n-ary classification for
     single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void predict(const cumlHandle& user_handle,
             const RandomForestClassifierF* forest, const float* input,
             int n_rows, int n_cols, int* predictions, int verbosity) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<float>> rf_classifier =
    std::make_shared<rfClassifier<float>>(forest->rf_params);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions,
                         forest, verbosity);
}

void predict(const cumlHandle& user_handle,
             const RandomForestClassifierD* forest, const double* input,
             int n_rows, int n_cols, int* predictions, int verbosity) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<double>> rf_classifier =
    std::make_shared<rfClassifier<double>>(forest->rf_params);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions,
                         forest, verbosity);
}
/** @} */

/**
 * @defgroup Random Forest Classification - Predict function
 * @brief Predict target feature for input data; n-ary classification for
     single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void predictGetAll(const cumlHandle& user_handle,
                   const RandomForestClassifierF* forest, const float* input,
                   int n_rows, int n_cols, int* predictions, int verbosity) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<float>> rf_classifier =
    std::make_shared<rfClassifier<float>>(forest->rf_params);
  rf_classifier->predictGetAll(user_handle, input, n_rows, n_cols, predictions,
                               forest, verbosity);
}

void predictGetAll(const cumlHandle& user_handle,
                   const RandomForestClassifierD* forest, const double* input,
                   int n_rows, int n_cols, int* predictions, int verbosity) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<double>> rf_classifier =
    std::make_shared<rfClassifier<double>>(forest->rf_params);
  rf_classifier->predictGetAll(user_handle, input, n_rows, n_cols, predictions,
                               forest, verbosity);
}
/** @} */

/**
 * @defgroup Random Forest Classification - Score function
 * @brief Compare predicted features validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @return RF_metrics struct with classification score (i.e., accuracy)
 * @{
 */
RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestClassifierF* forest, const int* ref_labels,
                 int n_rows, const int* predictions, int verbosity) {
  RF_metrics classification_score = rfClassifier<float>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity);
  return classification_score;
}

RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestClassifierD* forest, const int* ref_labels,
                 int n_rows, const int* predictions, int verbosity) {
  RF_metrics classification_score = rfClassifier<double>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity);
  return classification_score;
}

RF_params set_rf_class_obj(int max_depth, int max_leaves, float max_features,
                           int n_bins, int split_algo, int min_rows_per_node,
                           float min_impurity_decrease, bool bootstrap_features,
                           bool bootstrap, int n_trees, float rows_sample,
                           int seed, CRITERION split_criterion,
                           bool quantile_per_tree, int cfg_n_streams) {
  DecisionTree::DecisionTreeParams tree_params;
  DecisionTree::set_tree_params(
    tree_params, max_depth, max_leaves, max_features, n_bins, split_algo,
    min_rows_per_node, min_impurity_decrease, seed, bootstrap_features,
    split_criterion, quantile_per_tree);
  RF_params rf_params;
  set_all_rf_params(rf_params, n_trees, bootstrap, rows_sample, seed,
                    cfg_n_streams, tree_params);
  return rf_params;
}

/** @} */

/**
 * @defgroup Random Forest Regression - Fit function
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per
 *   training sample. Device pointer.
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void fit(const cumlHandle& user_handle, RandomForestRegressorF*& forest,
         float* input, int n_rows, int n_cols, float* labels,
         RF_params rf_params, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<float, float>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfRegressor<float>> rf_regressor =
    std::make_shared<rfRegressor<float>>(rf_params);
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels, forest);
}

void fit(const cumlHandle& user_handle, RandomForestRegressorD*& forest,
         double* input, int n_rows, int n_cols, double* labels,
         RF_params rf_params, int verbosity) {
  ML::Logger::get().setLevel(verbosity);
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<double, double>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfRegressor<double>> rf_regressor =
    std::make_shared<rfRegressor<double>>(rf_params);
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels, forest);
}
/** @} */

/**
 * @defgroup Random Forest Regression - Predict function
 * @brief Predict target feature for input data; regression for single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void predict(const cumlHandle& user_handle,
             const RandomForestRegressorF* forest, const float* input,
             int n_rows, int n_cols, float* predictions, int verbosity) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfRegressor<float>> rf_regressor =
    std::make_shared<rfRegressor<float>>(forest->rf_params);
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions, forest,
                        verbosity);
}

void predict(const cumlHandle& user_handle,
             const RandomForestRegressorD* forest, const double* input,
             int n_rows, int n_cols, double* predictions, int verbosity) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfRegressor<double>> rf_regressor =
    std::make_shared<rfRegressor<double>>(forest->rf_params);
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions, forest,
                        verbosity);
}
/** @} */

/**
 * @defgroup Random Forest Regression - Score function
 * @brief Predict target feature for input data and validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @return RF_metrics struct with regression score (i.e., mean absolute error,
 *   mean squared error, median absolute error)
 * @{
 */
RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestRegressorF* forest, const float* ref_labels,
                 int n_rows, const float* predictions, int verbosity) {
  RF_metrics regression_score = rfRegressor<float>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity);

  return regression_score;
}

RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestRegressorD* forest, const double* ref_labels,
                 int n_rows, const double* predictions, int verbosity) {
  RF_metrics regression_score = rfRegressor<double>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity);
  return regression_score;
}
/** @} */

// Functions' specializations
template void print_rf_summary<float, int>(
  const RandomForestClassifierF* forest);
template void print_rf_summary<double, int>(
  const RandomForestClassifierD* forest);
template void print_rf_summary<float, float>(
  const RandomForestRegressorF* forest);
template void print_rf_summary<double, double>(
  const RandomForestRegressorD* forest);

template void print_rf_detailed<float, int>(
  const RandomForestClassifierF* forest);
template void print_rf_detailed<double, int>(
  const RandomForestClassifierD* forest);
template void print_rf_detailed<float, float>(
  const RandomForestRegressorF* forest);
template void print_rf_detailed<double, double>(
  const RandomForestRegressorD* forest);

template void null_trees_ptr<float, int>(RandomForestClassifierF*& forest);
template void null_trees_ptr<double, int>(RandomForestClassifierD*& forest);
template void null_trees_ptr<float, float>(RandomForestRegressorF*& forest);
template void null_trees_ptr<double, double>(RandomForestRegressorD*& forest);

template void delete_rf_metadata<float, int>(RandomForestClassifierF* forest);
template void delete_rf_metadata<double, int>(RandomForestClassifierD* forest);
template void delete_rf_metadata<float, float>(RandomForestRegressorF* forest);
template void delete_rf_metadata<double, double>(
  RandomForestRegressorD* forest);

template void build_treelite_forest<float, int>(
  ModelHandle* model, const RandomForestMetaData<float, int>* forest,
  int num_features, int task_category);
template void build_treelite_forest<double, int>(
  ModelHandle* model, const RandomForestMetaData<double, int>* forest,
  int num_features, int task_category);
template void build_treelite_forest<float, float>(
  ModelHandle* model, const RandomForestMetaData<float, float>* forest,
  int num_features, int task_category);
template void build_treelite_forest<double, double>(
  ModelHandle* model, const RandomForestMetaData<double, double>* forest,
  int num_features, int task_category);
}  // End namespace ML
