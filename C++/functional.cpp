// others
#include <cmath>
// self
#include "functional.h"




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


// 根据 pos 和类的个数形成 one-hot 向量, 先只考虑一维度的 One-hot
vec_type one_hot(const int pos, const int len) {
	vec_type result(len, 0.0);
	result[pos] = 1.0;
	return result;
}