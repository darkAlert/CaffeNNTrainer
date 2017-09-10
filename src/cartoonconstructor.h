#ifndef CARTOONCONSTRUCTOR_H
#define CARTOONCONSTRUCTOR_H

#include <vector>
#include <array>
#include <opencv2/core.hpp>

namespace Landmarks68p {
enum : unsigned int {
    contour_index = 0,
    contour_count = 17,
    eyebrow_left_index = 17,
    eyebrow_right_index = 22,
    eyebrow_count = 5,
    nose_index = 27,
    nose_count = 9,
    eye_left_index = 36,
    eye_right_index = 42,
    eye_count = 6,
    mouth_index = 48,
    mouth_count = 20,
    total_count = 68
};
}

namespace FaceParts {
enum : unsigned int {
    brow_l = 0,
    brow_r = 1,
    eye_l,
    eye_r,
    nose,
    mouth,
    contour,
    size
};
}

class CartoonConstructor
{
private:
    //Vars:
    std::vector<std::vector<std::pair<cv::Rect2f, cv::Rect2f> > > scaffolds;                    //<photo, cartoon>
    std::vector<std::vector<std::pair<std::vector<float>, std::vector<float> > > > points;      //<photo, cartoon>

    //Methods:
    static void landmarks_to_rect(const std::vector<float> &landmarks68p, cv::Rect2f &rect, std::array<int, 2> range = {0,0});
    static void cut_landmarks(const std::vector<float> &landmarks, const cv::Rect2f &cut_rect,
                              std::vector<float> &cutted_landmarks, std::array<int, 2> range = {0,0});
    static void restore_landmarks(const std::pair<cv::Rect2f, cv::Rect2f> &scaffold,
                                  const std::pair<std::vector<float>, std::vector<float> > &points,
                                  std::vector<float> &photo_landmarks, std::vector<float> &cartoon_landmarks,
                                  const std::array<int, 2>  range = {0,Landmarks68p::total_count*2});

public:
    CartoonConstructor(const std::vector<std::pair<std::vector<float>, std::vector<float> > > &raw_landmarks);
    void generate(const uint scaffold_id, const std::vector<uint> &component_ids,
                  std::vector<float> &photo_landmarks, std::vector<float> &cartoon_landmarks);
    static void renormalize(std::vector<float> &landmarks);

    inline static void swap_landmarks(std::vector<float> &label, int src_first, int src_last, int dst_first, int dst_last) {
        int num1 = abs(src_last-src_first)+1;
        int num2 = abs(dst_last-dst_first)+1;
        assert(num1 == num2);
        for (int i = 0; i < num1; ++i) {
            std::swap(label[(src_first+i)*2],label[(dst_last-i)*2]);      //x
            std::swap(label[(src_first+i)*2+1],label[(dst_last-i)*2+1]);  //y
        }
    }

    //Getters:
    inline uint size() const {
        return scaffolds.size();
    }
};

#endif // CARTOONCONSTRUCTOR_H
