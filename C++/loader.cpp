// C 标准库
#include <io.h>
#include <direct.h>
// STL
#include <iostream>
#include <exception>
// 第三方库
#include "serialization.h"
// self
#include "scopeguard.hpp"
#include "loader.h"



std::pair< images_type, labels_type > load_from_txt(const std::string dataset_name) {
	// 找到对应的资源文件
	auto file_name = "./datasets/" + dataset_name + ".txt";
	// 定义文件输入流
	std::ifstream reader(file_name);
	YHL::ON_SCOPE_EXIT([&]{
		reader.close();
		std::cout << "file " << file_name << " is closed!\n";
	});
	if(!reader.is_open()) {
		std::cout << "read failed !\n"; 
		return std::make_pair(images_type(), labels_type());
	}
	else {
		// 申请内存存储图像和标签
		images_type images;
		labels_type labels;
		// 读取头部数据
		int train_size, input_size, signal, label;
		reader >> train_size >> input_size;
		std::cout << dataset_name << "  :  " << train_size << " images\ninput_size  :  " << input_size << "\n";
		for(auto cnt = 0;cnt < train_size; ++cnt) {
			// 读取一张图像的信号
			auto arr = std::vector<double>();
			for(auto i = 0;i < input_size; ++i) {
				reader >> signal;
				arr.emplace_back(double(signal));
			}
			images.emplace_back(arr);
			// 读入这张图象的标签
			reader >> label;
			labels.emplace_back(label);

			if((cnt + 1) % 1000 == 0)
				std::cout << cnt + 1 << "  is over...\n";
		}
		return std::make_pair(images, labels);
	}
}



Loader::Loader(const std::string& train_path, const std::string& test_path) 
			: train_path(train_path), test_path(test_path) {
	// 加载训练集
	if(access(train_path.c_str(), 0) not_eq 0)
		this->train_data = load_from_txt("train");
	else this->train_data = _cereal::load_pair<data_type>(train_path.c_str());
	// 加载测试集
	if(access(test_path.c_str(), 0) not_eq 0)
		this->test_data = load_from_txt("test");
	else this->test_data = _cereal::load_pair<data_type>(test_path.c_str());
	
	std::cout << "train  : \n\timages  " << this->train_data.first.size() << "\n\t" << this->train_data.second.size() << "\n";
	std::cout << "test  :  \n\timages  " << this->test_data.first.size() << "\n\t" << this->test_data.second.size() << "\n";
}

Loader::~Loader() noexcept {
	std::cout << "\n\nthe network is to be destroyed !";
	if(access(this->train_path.c_str(), 0) not_eq 0) 
		_cereal::dump_pair<data_type>("./datasets/train_data.binary", this->train_data);
	if(access(this->test_path.c_str(), 0) not_eq 0)
		_cereal::dump_pair<data_type>("./datasets/test_data.binary", this->test_data);
}


data_type& Loader::get_data(const std::string mode) {
	if(mode == "train") return this->train_data;
	else if(mode == "test") return this->test_data;
	return this->valid_data;
}


/**
 * opencv 处理 mnist
 * 涉及到读取 txt 文件
 * @time: 2020/09/23
 * @author: 刘畅
**/

/*
// STL 标准库
#include <direct.h>
#include <cmath>
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <exception> 
#include "quick_opencv.h"
// self
#include "scopeguard.hpp"



int main() {
	std::string dataset_name = "test";
	// 找到对应的资源文件
	auto file_name = "./" + dataset_name + ".txt";
	// 创建文件夹
	const std::string to_save = "datasets/" + dataset_name + "/";
	mkdir(to_save.c_str());
	// 定义文件输入流
	std::ifstream reader(file_name);
	YHL::ON_SCOPE_EXIT([&]{
		reader.close();
		std::cout << "file " << file_name << " is closed!\n";
	});
	if(!reader.is_open()) {
		std::cout << "read failed !\n";
	}
	else {
		// 读取头部数据
		int train_size, input_size, signal, label;
		reader >> train_size >> input_size;
		std::cout << "test  :  " << train_size << " images\ninput_size  :  " << input_size << "\n";
		for(auto cnt = 0;cnt < train_size; ++cnt) {
			// 读取一张图像的信号
			auto arr = std::vector<int>();
			for(auto i = 0;i < input_size; ++i) {
				reader >> signal;
				arr.emplace_back(signal);
			}
			// 将信号转化成图像
			auto _sz = int(std::sqrt(input_size));
			auto image = _cv2::Image(arr, true).reshape(0, _sz);
			auto result = 255 * _cv2::_type(image);
			// _cv2::imshow(result);
			// 读入这张图象的标签
			reader >> label;
			// 将图像根据标签保存到相应目录
			auto save_dir = to_save + std::to_string(label) + "/";
			mkdir(save_dir.c_str());
			_cv2::imwrite(save_dir + std::to_string(cnt + 1) + ".png", image * 255);

			if((cnt + 1) % 1000 == 0)
				std::cout << cnt + 1 << "  is over...";
		}
	}
	return 0;
}

*/