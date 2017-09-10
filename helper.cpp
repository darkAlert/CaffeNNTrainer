#include "helper.h"

//#include <opencv2/highgui.hpp>
#include <thread>
//#include <QCoreApplication>

namespace helper {

//void read_vector_from_image(const std::string &path_to_file, cv::Mat1f &vector)
//{
//    cv::Mat mat_c = cv::imread(path_to_file, 0);
//    vector = cv::Mat(1, mat_c.cols/4, CV_32FC1, mat_c.data);
//    vector = vector.clone();
//}

//void write_vector_to_image(const std::string &path_to_file, const cv::Mat1f &vector)
//{
//    cv::Mat mat_c = cv::Mat(1, vector.cols*4, CV_8UC1, vector.data);
//    cv::imwrite(path_to_file, mat_c);
//}

//double compute_distance(const cv::Mat1f &vector1, const cv::Mat1f &vector2)
//{
//    //L2-norm:
//    double L2_value1 = std::sqrt(cv::sum(vector1.mul(vector1))[0]);
//    double L2_value2 = std::sqrt(cv::sum(vector2.mul(vector2))[0]);
//    cv::Mat1f l2_norm_vector1 = vector1 / L2_value1;
//    cv::Mat1f l2_norm_vector2 = vector2 / L2_value2;

//    //Compute distance:
//    cv::Mat1f diff_vector = l2_norm_vector1 - l2_norm_vector2;
//    double distance = cv::sum(diff_vector.mul(diff_vector))[0];

//    return distance;
//}

void little_sleep(std::chrono::microseconds us)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + us;
    do {
        std::this_thread::yield();
    } while (std::chrono::high_resolution_clock::now() < end);
}

//std::string absolutePath(std::string path)
//{
//    return QCoreApplication::applicationDirPath().toStdString() + "/" + path;
//}

}


