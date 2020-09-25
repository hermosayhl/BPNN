// C
#include <io.h>
#include <direct.h>
// STL
#include <ctime>
#include <cmath>
#include <random>
#include <chrono>
#include <assert.h>
#include <numeric>
#include <iostream>
#include <algorithm>
// 第三方库
#include "serialization.h"
// self
#include "loader.h"



namespace {
	
	using vec_type = std::vector<double>;
	using weights_type = std::vector<vec_type>;
	vec_type null;

	
	template<typename T>
	void display(const T& arr) {
		for(auto& it : arr)
			std::cout << it << " ";
		std::cout << "\n";
	}

	// 按照一定的分布生成数据, 返回值类型还不确定
	vec_type random_distribution(const int len) {
		// 设置随机种子
		auto seed = std::rand();
	    std::default_random_engine generator(seed);
	    std::normal_distribution<double> distribution(0.0, 1.0);
	    // 开始采样 len 个数据
		vec_type result(len, 0);
		for(int i = 0;i < len; ++i)
			result[i] = distribution(generator) / std::sqrt(len); // / std::sqrt(len)
		return result;
	}

	// 将两个 vector 同步打乱, assert 长度是否一致
	template<typename T>
	void co_shuffle(T& lhs, T& rhs) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	    std::shuffle(lhs.begin (), lhs.end(), std::default_random_engine(seed));
	    std::shuffle(rhs.begin (), rhs.end(), std::default_random_engine(seed));
	}

	// 根据 pos 和类的个数形成 one-hot 向量, 先只考虑一维度的 One-hot
	vec_type one_hot(const int pos, const int len) {
		vec_type result(len, 0.0);
		result[pos] = 1.0;
		return result;
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
			// 这里如果是指针可以浅拷贝就好了, C 里方便; std::vector 都杜绝浅拷贝了
			this->lhs = lhs;
			this->rhs = rhs;
			const int len = lhs.size();
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

}

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

	void load(const std::string& path) {
		assert(path.size() > 0);
		std::cout << "根据路径  " << path << "\n";
		for(int i = 0;i < this->layer_size; ++i) {
			auto layer_name = path + "_layer_" + std::to_string(i) + "_weights.binary";
			std::cout << "读取文件  " << layer_name << "\n";
			this->layers[i].weights = _cereal::load<weights_type>(layer_name);
			this->layers[i].bias = _cereal::load<vec_type>(layer_name.replace(layer_name.find("weights"), 7, "bias"));
		}
		std::cout << "预训练权重加载成功 !\n";
	}

	void save(const int epoch, const std::string& checkpoint_path="./checkpoints/") {
		// 如果不存在当前文件夹, 创建
		if(access(checkpoint_path.c_str(), 0) not_eq 0) {
			mkdir(checkpoint_path.c_str());
		}
		const std::string save_dir = checkpoint_path + "epoch_" + std::to_string(epoch);
		if(access(save_dir.c_str(), 0) not_eq 0)
			mkdir(save_dir.c_str());
		const std::string base_name = save_dir + "/" + "epoch_" + std::to_string(epoch);
		std::cout << "保存到文件夹  " << checkpoint_path << "\n";
		// 开始分开保存每一层的权重
		for(int i = 0;i < this->layer_size; ++i) {
			auto layer_name = base_name + "_layer_" + std::to_string(i) + "_weights.binary";
			_cereal::dump<weights_type>(layer_name, this->layers[i].weights);
			_cereal::dump<vec_type>(layer_name.replace(layer_name.find("weights"), 7, "bias"), this->layers[i].bias);
			std::cout << "\t文件  " << layer_name << "\n";
		}
	} 

	double score(const images_type& x, const labels_type& y) {
		// 这里最好检查一下 x, y 的 size 是否匹配
		int cnt = 0;
		const int len = y.size();
		// 获取估计值, 逐一比较 
		for(int i = 0;i < len; ++i)
			if(this->recognize(x[i]) == y[i]) ++cnt;
		return static_cast<double>(cnt * 1.0 / len);
	}

	int recognize(const vec_type& inpt) {
		auto output = this->operator()(inpt);
		return std::max_element(output.begin(), output.end()) - output.begin();
	}

	vec_type operator()(const vec_type& inpt) {
		vec_type output = inpt;
		for(int i = 0;i < this->layer_size; ++i) {
			output = this->layers[i](output);
			output = this->activations[i](output);
		}
		return output;
	}

	void backward(const double learning_rate=1e-1) {
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

	BPNN(const std::vector<int>& dimensions): 
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


class Solver {
public:
	void train(const int epoches=1, const double learning_rate=1e-1, const bool shuffle=false) {
		// 获取数据
		auto &train_data = this->loader.get_data("train");
		const int out_size = this->bpnn.neurons[this->bpnn.layer_size];
		// 记录损失
		const int log_interval = 1000;
		loss_recoder recoder(log_interval);
		// 开始训练
		for(auto epoch = 0;epoch < epoches; ++epoch) {
			auto &images = train_data.first;
			auto &labels = train_data.second;
			const int len = images.size();
			std::cout << "一共要训练  " << len  << "张图象\n";
			for(int i = 0;i < len; ++i) {
				// 转化为 one-hot 标签
				auto label = one_hot(labels[i], out_size); 
				// 输出未归一化的 10 个数字概率
				auto output = this->bpnn(images[i]);
				// 衡量损失
				auto loss_value = this->bpnn.loss_fn(output, label);
				recoder.log(loss_value);
				// 根据损失回传梯度, 参数 learning_rate 学习率
				this->bpnn.backward();
				// 输出一些信息
				if((i + 1) % log_interval == 0) 
					std::cout << i + 1 << "  is over...\tloss = " << recoder.log_output() << std::endl;
			}
			// 训练之后测试一下
			this->test();
			// 保存当前模型
			this->bpnn.save(epoch);
		}
	}

	void test(const std::string& path="") {
		if(path.size() not_eq 0) 
			this->bpnn.load(path);
		auto& test_data = this->loader.get_data("test");
		const double score = this->bpnn.score(test_data.first, test_data.second);
		std::cout << "accuracy===>  " << score << "\n";
	}

	Solver() = default;
	~Solver() noexcept = default;
private:
	BPNN bpnn = BPNN({784, 100, 10});
	Loader loader = Loader("./datasets/train_data.binary", "./datasets/test_data.binary");
};



int main() {
	std::cout << "compile success !" << std::endl;

	Solver solver;
	// solver.train();
	solver.test("./checkpoints/epoch_0/epoch_0");
	return 0;
	// inline
	// 只支持随机梯度下降
	// 矩阵的长度检测, 这个非常重要
	// friend 友类很重要
}
