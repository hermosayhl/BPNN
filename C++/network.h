#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include "functional.h"


/*
	auto bpnn = BPNN({784, 100, 10});
	auto inpt = random_distribution(784);
	auto output = bpnn(inpt);
	auto label = vec_type({0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
	auto loss_value = bpnn.loss_fn(output, label);
	bpnn.backward();
*/



class BPNN {
public:

	void load(const std::string& path);

	void save(const int epoch, const double accuracy, const std::string& checkpoint_path="./checkpoints/");

	double score(const images_type& x, const labels_type& y);

	int recognize(const vec_type& inpt);

	vec_type operator()(const vec_type& inpt);

	void backward(const double learning_rate=1e-1);

	BPNN(const std::vector<int>& dimensions);
	~BPNN() noexcept = default;

public:
	MseLoss loss_fn;
	const std::vector<int> neurons;
	const int layer_size;
private:
	std::vector<Linear> layers;
	std::vector<Sigmoid> activations;
	std::vector<vec_type> delta;
};



#endif