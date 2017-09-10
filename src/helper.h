#ifndef HELPER_H
#define HELPER_H

//#include <opencv2/core.hpp>
#include <chrono>

namespace helper {

//void read_vector_from_image(const std::string &path_to_file, cv::Mat1f &vector);
//void write_vector_to_image(const std::string &path_to_file, const cv::Mat1f &vector);
//double compute_distance(const cv::Mat1f &vector1, const cv::Mat1f &vector2);
void little_sleep(std::chrono::microseconds us);
//std::string absolutePath(std::string path);
}

#endif // HELPER_H
