/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuML.hpp>
#include "randomforest.h"
#include "decisiontree/decisiontree.h"

namespace ML{

struct RF_metrics {
	float accuracy;

	RF_metrics(float cfg_accuracy);
	void print();
};

template <class T>
class rfClassifier : public rf<T> {
	public:

	rfClassifier(RF_params cfg_rf_params);

	void fit(const cumlHandle& user_handle, T * input, int n_rows, int n_cols, int * labels, int n_unique_labels);
	void predict(const cumlHandle& user_handle, const T * input, int n_rows, int n_cols, int * predictions, bool verbose=false) const;
	RF_metrics cross_validate(const cumlHandle& user_handle, const T * input, const int * ref_labels, int n_rows, int n_cols, int * predictions, bool verbose=false) const;

};


void fit(const cumlHandle& user_handle, rfClassifier<float> * rf_classifier, float * input, int n_rows, int n_cols, int * labels, int n_unique_labels);
void fit(const cumlHandle& user_handle, rfClassifier<double> * rf_classifier, double * input, int n_rows, int n_cols, int * labels, int n_unique_labels);

void predict(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, int n_rows, int n_cols, int * predictions, bool verbose=false);
void predict(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double * input, int n_rows, int n_cols, int * predictions, bool verbose=false);

RF_metrics cross_validate(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, const int * ref_labels,
							int n_rows, int n_cols, int * predictions, bool verbose=false);
RF_metrics cross_validate(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double * input, const int * ref_labels,
							int n_rows, int n_cols, int * predictions, bool verbose=false);

};

