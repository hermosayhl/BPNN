#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include <vector>
#include <random>
#include <chrono>
#include <assert.h>
#include <iostream>
#include <algorithm>

// 定义很多简易的修饰符
using vec_type = std::vector<double>;
using weights_type = std::vector<vec_type>;
using images_type = std::vector< std::vector<double> >; // 和 weights_type 一样, 但是代表的变量含义不一样
using labels_type = std::vector<int>;




// 根据 pos 和类的个数形成 one-hot 向量, 先只考虑一维度的 One-hot
vec_type one_hot(const int pos, const int len);

// 按照一定的分布生成数据, 返回值类型还不确定
vec_type random_distribution(const int len);




// 打印某种数据类型的一维迭代器
template<typename T>
inline void display(const T& arr) {
	for(auto& it : arr)
		std::cout << it << " ";
	std::cout << "\n";
}




// 将两个 vector 同步打乱, assert 长度是否一致
template<typename T>
inline void co_shuffle(T& lhs, T& rhs) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	assert(lhs.size() == rhs.size());
    std::shuffle(lhs.begin (), lhs.end(), std::default_random_engine(seed));
    std::shuffle(rhs.begin (), rhs.end(), std::default_random_engine(seed));
}



// 每隔 interval 输出一次平均损失值
class loss_recoder {
public:
	loss_recoder(const int interval): interval(interval) {}
	void log(const double loss_value) {
		this->loss_sum += loss_value;
	}
	double log_output() {
		const double loss_mean = this->loss_sum / this->interval;
		this->loss_sum = 0;
		return loss_mean;
	}
private:
	const int interval;
	double loss_sum;
};





/*  Sigmoid 激活函数
		sigmoid(x) = 1 / (1 + exp(-x))
		sigmoid`(x) = x * (1 - x)
    示例
		auto activation = Sigmoid();
		vec_type output({0.5, 0, 0.1, -0.5});
		auto result = activation(output);
		display(result);
		display(activation.backward());
	这里可以使用虚函数, 回顾一下 C++ 的多态。
*/
class Sigmoid {
public:
	// 重载对象调用方法, 等效于 python 中的 __call__()
	vec_type operator()(const vec_type& inpt) {
		this->output.assign(inpt.begin(), inpt.end());
		for(auto &x : this->output) 
			x = 1.0 / (1.0 + std::exp(-x));
		return this->output;
	}
	vec_type backward() {
		for(auto &x: this->output) 
			x = x * (1.0 - x);
		return this->output; 
	}
private:
	vec_type output;  
};


/*  MSE 损失函数
		mse(l, r) = 0.5 * (l - r) ^ 2
		mse`(l, r) = l - r
	示例(forward 的顺序不可以反)
		auto loss_fn = MseLoss();
		auto lhs = vec_type({0.8, -0.8, 0.1, 0});
		auto rhs = vec_type({0, 0.9, -0.2, 0.7});
		auto loss_value = loss_fn(lhs, rhs);
		display(loss_value);
		display(loss_fn.backward());
*/
class MseLoss {
public:
	double operator()(const vec_type& lhs, const vec_type& rhs) {
		const int len = lhs.size();
		assert(len == static_cast<int>(rhs.size()));
		this->lhs = lhs;   // 这里如果是指针可以浅拷贝就好了, C 里方便; std::vector 都杜绝浅拷贝了
		this->rhs = rhs;
		this->output.assign(len, 0.0);
		for(int i = 0;i < len; ++i)
			this->output[i] = 0.5 * std::pow(lhs[i] - rhs[i], 2);
		return std::accumulate(this->output.begin(), this->output.end(), 0.0);
	}
	vec_type backward() {
		const int len = this->output.size();
		for(int i = 0;i < len; ++i)
			this->output[i] = this->lhs[i] - this->rhs[i];
		return this->output;
	}
private:
	vec_type lhs, rhs;
	vec_type output;
};


/*  是啊, 这里如果用拷贝, 开销就太大了!!! 还是指针靠谱 ?  对啊啊啊啊啊 引用!!!!!! &
	std::srand(std::time(nullptr));

	auto layer = Linear(2, 3);
	auto inpt = vec_type({1.5, 2.5});
	auto result = layer(inpt);
	std::cout << "前向结果是  :  \n";
	display(result);
*/
class Linear {
public:
	vec_type operator()(const vec_type& inpt) {
		// 检查维度是否匹配, 太坑了
		assert(inpt.size() == this->weights.size());

		this->inpt = inpt;
		vec_type output(this->out_dim, 0.0);
		for(int i = 0;i < this->out_dim; ++i) {
			double temp = 0;
			for(int j = 0;j < this->in_dim; ++j)
				temp += inpt[j] * this->weights[j][i];
			output[i] = temp + this->bias[i];
		}
		return output;
	}

	vec_type backward() const {
		return this->inpt;
	}

	Linear(const int in_dim, const int out_dim) :
			in_dim(in_dim), 
			out_dim(out_dim), 
			inpt(in_dim, 0.0),
			bias(random_distribution(out_dim)){
		// 初始化权重
		for(int i = 0;i < in_dim; ++i) 
			this->weights.emplace_back(random_distribution(this->out_dim));
	}
	~Linear() noexcept = default;
public:
	const int in_dim, out_dim;
	// friend class BPNN;
// private:
	// 记录历史信息
	vec_type inpt;
	// 要学习的参数
	weights_type weights;
	vec_type bias;
};


#endif