#pragma once
#include "algo_helper.h"


namespace ML {
namespace DecisionTree {
struct DecisionTreeParams {
	/**
	 * Maximum tree depth. Unlimited (e.g., until leaves are pure), if -1.
	 */
	int max_depth = -1;
	/**
	 * Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
	 */
	int max_leaves = -1;
	/**
	 * Ratio of number of features (columns) to consider per node split.
	   TODO SKL's default is sqrt(n_cols)
	 */
	float max_features = 1.0;
	/**
	 * Number of bins used by the split algorithm.
	 */
	int n_bins = 8;
	/**
	 * The split algorithm: HIST or GLOBAL_QUANTILE.
	 */
	int split_algo = SPLIT_ALGO::HIST;
	/**
	 * The minimum number of samples (rows) needed to split a node.
	 */
	int min_rows_per_node = 2;
	/**
	 * Wheather to bootstarp columns with or without replacement
	 */
	bool bootstrap_features =  false;
	
	DecisionTreeParams();
	DecisionTreeParams(int cfg_max_depth, int cfg_max_leaves, float cfg_max_features, int cfg_n_bins, int cfg_split_aglo, int cfg_min_rows_per_node, bool cfg_bootstrap_features);
	void validity_check() const;
	void print() const;
};

template<class T>
class DecisionTreeClassifier {

private:
	int split_algo;
	TreeNode<T> *root = nullptr;
	int nbins;
	DataInfo dinfo;
	int treedepth;
	int depth_counter = 0;
	int maxleaves;
	int leaf_counter = 0;
	std::vector<std::shared_ptr<TemporaryMemory<T>>> tempmem;
	size_t total_temp_mem;
	const int MAXSTREAMS = 1;
	size_t max_shared_mem;
	size_t shmem_used = 0;
	int n_unique_labels = -1; // number of unique labels in dataset
	double construct_time;
	int min_rows_per_node;
	bool bootstrap_features;
	std::vector<int> feature_selector;
	
public:
	// Expects column major T dataset, integer labels
	// data, labels are both device ptr.
	// Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
	void fit(const ML::cumlHandle& handle, T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids,
			const int n_sampled_rows, const int unique_labels, DecisionTreeParams tree_params);

	/* Predict labels for n_rows rows, with n_cols features each, for a given tree. rows in row-major format. */
	void predict(const ML::cumlHandle& handle, const T * rows, const int n_rows, const int n_cols, int* predictions, bool verbose=false) const;

	// Printing utility for high level tree info.
	void print_tree_summary() const;

	// Printing utility for debug and looking at nodes and leaves.
	void print() const;

private:
	// Same as above fit, but planting is better for a tree then fitting.
	void plant(const cumlHandle_impl& handle, T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels,
		   int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8, int split_algo_flag = SPLIT_ALGO::HIST, int cfg_min_rows_per_node=2, bool cfg_bootstrap_features=false);

	TreeNode<T> * grow_tree(T *data, const float colper, int *labels, int depth, unsigned int* rowids, const int n_sampled_rows, GiniInfo prev_split_info);

	/* depth is used to distinguish between root and other tree nodes for computations */
	void find_best_fruit_all(T *data, int *labels, const float colper, GiniQuestion<T> & ques, float& gain, unsigned int* rowids,
							const int n_sampled_rows, GiniInfo split_info[3], int depth);
	void split_branch(T *data, GiniQuestion<T> & ques, const int n_sampled_rows, int& nrowsleft, int& nrowsright, unsigned int* rowids);
	void classify_all(const T * rows, const int n_rows, const int n_cols, int* preds, bool verbose=false) const;
	int classify(const T * row, const TreeNode<T> * const node, bool verbose=false) const;
	void print_node(const std::string& prefix, const TreeNode<T>* const node, bool isLeft) const;
}; // End DecisionTree Class
};
};