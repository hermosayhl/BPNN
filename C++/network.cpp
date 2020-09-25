// others
#include <io.h>
#include <direct.h>
#include <assert.h>
#include <numeric>
#include <fstream>
// self
#include "network.h"
#include "scopeguard.hpp"




void BPNN::load(const std::string& path) {
	assert(static_cast<int>(path.size()) > 0);
	std::cout << "根据路径  " << path << "\n";
	// 设置读写流
	std::ifstream reader(path.c_str());
	assert(reader);
	YHL::ON_SCOPE_EXIT([&]{
		reader.close(); std::cout << "权重加载成功 !\n";
	});
	for(int i = 0;i < this->layer_size; ++i) {
		for(int j = 0;j < this->neurons[i]; ++j) 
			for(int k = 0;k < this->neurons[i + 1]; ++k)
				reader >> this->layers[i].weights[j][k];
		for(int j = 0;j < this->neurons[i + 1]; ++j)
			reader >> this->layers[i].bias[j];
	}
}




void BPNN::save(const int epoch, const double accuracy, const std::string& checkpoint_path) {
	// 如果不存在当前文件夹, 创建
	if(access(checkpoint_path.c_str(), 0) not_eq 0) {
		mkdir(checkpoint_path.c_str());
	}
	// 直接写到 txt 文件
	auto file_name = "epoch_" + std::to_string(epoch) + "_accuracy_" + std::to_string(accuracy) + "_weights.txt";
	std::ofstream writer((checkpoint_path + file_name).c_str());
	YHL::ON_SCOPE_EXIT([&]{
		writer.close(); std::cout << "权重已经写入文件  " << checkpoint_path + file_name << "\n";
	});
	// 开始写入文件
	for(int i = 0;i < this->layer_size; ++i) {
		// weights
		for(int j = 0;j < this->neurons[i]; ++j) {
			for(int k = 0;k < this->neurons[i + 1]; ++k)
				writer << this->layers[i].weights[j][k] << " ";
			writer << "\n";
		}
		// bias
		for(int j = 0;j < this->neurons[i + 1]; ++j) 
			writer << this->layers[i].bias[j] << " ";
		writer << "\n";
	}
} 



double BPNN::score(const images_type& x, const labels_type& y) {
	// 这里最好检查一下 x, y 的 size 是否匹配
	int cnt = 0;
	const int len = y.size();
	// 获取估计值, 逐一比较 
	for(int i = 0;i < len; ++i)
		if(this->recognize(x[i]) == y[i]) ++cnt;
	return static_cast<double>(cnt * 1.0 / len);
}




int BPNN::recognize(const vec_type& inpt) {
	auto output = this->operator()(inpt);
	return std::max_element(output.begin(), output.end()) - output.begin();
}




vec_type BPNN::operator()(const vec_type& inpt) {
	vec_type output = inpt;
	for(int i = 0;i < this->layer_size; ++i) {
		output = this->layers[i](output);
		output = this->activations[i](output);
	}
	return output;
}




void BPNN::backward(const double learning_rate) {
	// 先计算梯度
	for(int i = this->layer_size - 1;i >= 0; --i) {
		// 获取激活函数的梯度向量
		auto activation_error = this->activations[i].backward();
		// 初始化梯度向量
		const int delta_size = this->neurons[i + 1];
		// error 表示误差, 算激活层梯度之前
		vec_type error(delta_size, 0.0);  
		// 如果是最后一层
		if(i == this->layer_size - 1) {
			error = this->loss_fn.backward();
		}
		else {
			// 中间层, 每一个权重都会影响到下一层的所有神经元; 梯度反向更新时, 需要求和
            auto &weights = this->layers[i + 1].weights; // 100 * 10
			for(int j = 0;j < delta_size; ++j) {
				double sum_value = 0.0;
				for(int k = 0;k < this->neurons[i + 2]; ++k) 
					sum_value += weights[j][k] * this->delta[i + 1][k];
				error[j] = sum_value;
			}
		}
		// 算出了 error, 准备算上激活层的梯度
		for(int j = 0;j < delta_size; ++j) 
			this->delta[i][j] = error[j] * activation_error[j];
	}
	// ------------------------------------------------------------------------------------------------------
	// 开始进行梯度更新
	for(int i = 0;i < this->layer_size; ++i) {
		auto &weights = this->layers[i].weights; // 784 * 100   
		auto error_linear = this->layers[i].backward();  // 784 之前求梯度都只是到了 wx + b 的输出层, 还有个 linear 层的 backward
		// 每层的 linear 层
		for(int j = 0;j < this->neurons[i]; ++j) {
			for(int k = 0;k < this->neurons[i + 1]; ++k) {
				double residual = error_linear[j] * this->delta[i][k];
				weights[j][k] -= learning_rate * residual;
			}
		}
		for(int j = 0;j < this->neurons[i + 1]; ++j) this->layers[i].bias[j] -= learning_rate * this->delta[i][j];
	}
}


BPNN::BPNN(const std::vector<int>& dimensions): 
		neurons(dimensions), 
		layer_size(dimensions.size() - 1), 
		activations(dimensions.size() - 1, Sigmoid()) {
	// 这里开始添加网络层
	for(int i = 0;i < this->layer_size; ++i) {
		this->layers.emplace_back(Linear(this->neurons[i], this->neurons[i + 1]));
		std::cout << "layer  :  " << this->layers[i].in_dim << "\t" << this->layers[i].out_dim << "\n";
	}
	// 初始化梯度历史
	for(int i = 0;i < this->layer_size; ++i)
		this->delta.emplace_back(vec_type(this->neurons[i + 1], 0.0));

}