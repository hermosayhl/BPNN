#ifndef LOADER_H
#define LOADER_H
#include <vector>
#include <string>


using images_type = std::vector< std::vector<double> >;
using labels_type = std::vector<int>;
using data_type = std::pair< images_type, labels_type >;

/*
	auto loader = Loader("./datasets/train_data.binary", "./datasets/test_data.binary");
*/



// 从文件读取 mnist
std::pair< images_type, labels_type > load_from_txt(const std::string dataset_name);



class Loader {
public:
	Loader(const std::string& train_path, const std::string& test_path);
	~Loader() noexcept;
	data_type& get_data(const std::string mode);

private:
	// 路径
	std::string train_path;
	std::string test_path;
	// 数据
	data_type train_data;
	data_type test_data;
	data_type valid_data;
};



#endif



