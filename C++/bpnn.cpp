// C
// STL
// 第三方库
// -------------------------------
// self
#include "loader.h"
#include "network.h"




class Solver {
	// 目前只支持 SGD 随机梯度下降
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
				this->bpnn.backward(6 * 1e-2);
				// 输出一些信息
				if((i + 1) % log_interval == 0) 
					std::cout << i + 1 << "  is over...\tloss = " << recoder.log_output() << std::endl;
			}
			// 训练之后测试一下
			auto accuracy = this->test();
			// 保存当前模型
			this->bpnn.save(epoch, accuracy);
			// 可以打乱一下
			co_shuffle(images, labels);
		}
	}

	double test(const std::string& path="") {
		if(path.size() not_eq 0) 
			this->bpnn.load(path);
		auto& test_data = this->loader.get_data("test");
		const double score = this->bpnn.score(test_data.first, test_data.second);
		std::cout << "accuracy===>  " << score << "\n";
		return score;
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
	solver.train(10);
	// solver.test("./checkpoints/epoch_0_accuracy_0.933000_weights.txt");
	return 0;
}
