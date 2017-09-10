#include "cartoonconstructor.h"

CartoonConstructor::CartoonConstructor(const std::vector<std::pair<std::vector<float>, std::vector<float> > > &raw_landmarks)
{
    assert(raw_landmarks.empty() == false);
    scaffolds.resize(raw_landmarks.size());
    points.resize(raw_landmarks.size());

    for (uint i = 0; i < raw_landmarks.size(); ++i)
    {
        assert(raw_landmarks[i].first.size() == Landmarks68p::total_count*2);
        assert(raw_landmarks[i].second.size() == Landmarks68p::total_count*2);

        /// Make scaffold from given landmarks ///
        scaffolds[i].resize(FaceParts::size);
        cv::Rect2f photo_bbox, cartoon_bbox;
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::eye_left_index*2, Landmarks68p::eye_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::eye_left_index*2, Landmarks68p::eye_count*2});
        scaffolds[i][FaceParts::eye_l] = std::make_pair(photo_bbox, cartoon_bbox);

        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::eye_right_index*2, Landmarks68p::eye_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::eye_right_index*2, Landmarks68p::eye_count*2});
        scaffolds[i][FaceParts::eye_r] = std::make_pair(photo_bbox, cartoon_bbox);

        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::eyebrow_left_index*2, Landmarks68p::eyebrow_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::eyebrow_left_index*2, Landmarks68p::eyebrow_count*2});
        scaffolds[i][FaceParts::brow_l] = std::make_pair(photo_bbox, cartoon_bbox);

        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::eyebrow_right_index*2, Landmarks68p::eyebrow_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::eyebrow_right_index*2, Landmarks68p::eyebrow_count*2});
        scaffolds[i][FaceParts::brow_r] = std::make_pair(photo_bbox, cartoon_bbox);

        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::nose_index*2, Landmarks68p::nose_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::nose_index*2, Landmarks68p::nose_count*2});
        scaffolds[i][FaceParts::nose] = std::make_pair(photo_bbox, cartoon_bbox);

        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::mouth_index*2, Landmarks68p::mouth_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::mouth_index*2, Landmarks68p::mouth_count*2});
        scaffolds[i][FaceParts::mouth] = std::make_pair(photo_bbox, cartoon_bbox);

        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].first, photo_bbox, {Landmarks68p::contour_index*2, Landmarks68p::contour_count*2});
        CartoonConstructor::landmarks_to_rect(raw_landmarks[i].second, cartoon_bbox, {Landmarks68p::contour_index*2, Landmarks68p::contour_count*2});
        scaffolds[i][FaceParts::contour] = std::make_pair(photo_bbox, cartoon_bbox);


        /// Cut points from given landmarks ///
        points[i].resize(FaceParts::size);
        std::vector<float> cutted_photo_points, cutted_cartoon_points;
        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::eye_l].first, cutted_photo_points,
                                          {Landmarks68p::eye_left_index*2, Landmarks68p::eye_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::eye_l].second, cutted_cartoon_points,
                                          {Landmarks68p::eye_left_index*2, Landmarks68p::eye_count*2});
        points[i][FaceParts::eye_l] = std::make_pair(cutted_photo_points, cutted_cartoon_points);

        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::eye_r].first, cutted_photo_points,
                                          {Landmarks68p::eye_right_index*2, Landmarks68p::eye_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::eye_r].second, cutted_cartoon_points,
                                          {Landmarks68p::eye_right_index*2, Landmarks68p::eye_count*2});
        points[i][FaceParts::eye_r] = std::make_pair(cutted_photo_points, cutted_cartoon_points);

        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::brow_l].first, cutted_photo_points,
                                          {Landmarks68p::eyebrow_left_index*2, Landmarks68p::eyebrow_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::brow_l].second, cutted_cartoon_points,
                                          {Landmarks68p::eyebrow_left_index*2, Landmarks68p::eyebrow_count*2});
        points[i][FaceParts::brow_l] = std::make_pair(cutted_photo_points, cutted_cartoon_points);

        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::brow_r].first, cutted_photo_points,
                                          {Landmarks68p::eyebrow_right_index*2, Landmarks68p::eyebrow_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::brow_r].second, cutted_cartoon_points,
                                          {Landmarks68p::eyebrow_right_index*2, Landmarks68p::eyebrow_count*2});
        points[i][FaceParts::brow_r] = std::make_pair(cutted_photo_points, cutted_cartoon_points);

        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::nose].first, cutted_photo_points,
                                          {Landmarks68p::nose_index*2, Landmarks68p::nose_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::nose].second, cutted_cartoon_points,
                                          {Landmarks68p::nose_index*2, Landmarks68p::nose_count*2});
        points[i][FaceParts::nose] = std::make_pair(cutted_photo_points, cutted_cartoon_points);

        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::mouth].first, cutted_photo_points,
                                          {Landmarks68p::mouth_index*2, Landmarks68p::mouth_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::mouth].second, cutted_cartoon_points,
                                          {Landmarks68p::mouth_index*2, Landmarks68p::mouth_count*2});
        points[i][FaceParts::mouth] = std::make_pair(cutted_photo_points, cutted_cartoon_points);

        CartoonConstructor::cut_landmarks(raw_landmarks[i].first, scaffolds[i][FaceParts::contour].first, cutted_photo_points,
                                          {Landmarks68p::contour_index*2, Landmarks68p::contour_count*2});
        CartoonConstructor::cut_landmarks(raw_landmarks[i].second, scaffolds[i][FaceParts::contour].second, cutted_cartoon_points,
                                          {Landmarks68p::contour_index*2, Landmarks68p::contour_count*2});
        points[i][FaceParts::contour] = std::make_pair(cutted_photo_points, cutted_cartoon_points);
    }
}

void CartoonConstructor::generate(const uint scaffold_id, const std::vector<uint> &component_ids,
                                  std::vector<float> &photo_landmarks, std::vector<float> &cartoon_landmarks)
{
    assert(scaffold_id < scaffolds.size());
    photo_landmarks.resize(Landmarks68p::total_count*2);
    cartoon_landmarks.resize(Landmarks68p::total_count*2);

    /// Restore landmarks ///
    uint comp_id;
    uint face_part;
    std::array<int, 2> range;

    face_part = FaceParts::nose;
    range = {Landmarks68p::nose_index*2, (Landmarks68p::nose_index+Landmarks68p::nose_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);

    face_part = FaceParts::mouth;
    range = {Landmarks68p::mouth_index*2, (Landmarks68p::mouth_index+Landmarks68p::mouth_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);

    face_part = FaceParts::eye_l;
    range = {Landmarks68p::eye_left_index*2, (Landmarks68p::eye_left_index+Landmarks68p::eye_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);

    face_part = FaceParts::eye_r;
    range = {Landmarks68p::eye_right_index*2, (Landmarks68p::eye_right_index+Landmarks68p::eye_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);

    face_part = FaceParts::brow_l;
    range = {Landmarks68p::eyebrow_left_index*2, (Landmarks68p::eyebrow_left_index+Landmarks68p::eyebrow_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);

    face_part = FaceParts::brow_r;
    range = {Landmarks68p::eyebrow_right_index*2, (Landmarks68p::eyebrow_right_index+Landmarks68p::eyebrow_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);

    face_part = FaceParts::contour;
    range = {Landmarks68p::contour_index*2, (Landmarks68p::contour_index+Landmarks68p::contour_count)*2};
    comp_id = component_ids[face_part];
    restore_landmarks(scaffolds[scaffold_id][face_part], points[comp_id][face_part], photo_landmarks, cartoon_landmarks, range);
}

void CartoonConstructor::restore_landmarks(const std::pair<cv::Rect2f, cv::Rect2f> &scaffold,
                                           const std::pair<std::vector<float>, std::vector<float> > &points,
                                           std::vector<float> &photo_landmarks, std::vector<float> &cartoon_landmarks,
                                           const std::array<int, 2> range)
{
    for (int i = range[0], j = 0; i < range[1]; i+=2, j+=2)
    {
        //Photo points:
        photo_landmarks[i] = scaffold.first.x + (points.first[j]*scaffold.first.width);
        photo_landmarks[i+1] = scaffold.first.y + (points.first[j+1]*scaffold.first.height);
        //Cartoon points:
        cartoon_landmarks[i] = scaffold.second.x + (points.second[j]*scaffold.second.width);
        cartoon_landmarks[i+1] = scaffold.second.y + (points.second[j+1]*scaffold.second.height);
    }
}

void CartoonConstructor::landmarks_to_rect(const std::vector<float> &landmarks68p, cv::Rect2f &rect, std::array<int, 2> range)
{
    if (range[1] == 0) range[1] = landmarks68p.size();
    else range[1] = range[0] + range[1];

    float min_x = landmarks68p[range[0]], max_x = landmarks68p[range[0]];
    float min_y = landmarks68p[range[0]+1], max_y = landmarks68p[range[0]+1];

    for (int i = range[0]; i < range[1]; i+=2) {
        if (landmarks68p[i] < min_x) min_x = landmarks68p[i];
        if (landmarks68p[i+1] < min_y) min_y = landmarks68p[i+1];
        if (landmarks68p[i] > max_x) max_x = landmarks68p[i];
        if (landmarks68p[i+1] > max_y) max_y = landmarks68p[i+1];
    }

    rect.x = min_x;
    rect.y = min_y;
    rect.width = max_x - min_x;
    rect.height = max_y - min_y;
}


void CartoonConstructor::cut_landmarks(const std::vector<float> &landmarks, const cv::Rect2f &cut_rect,
                   std::vector<float> &cutted_landmarks, std::array<int, 2> range)
{
    if (range[1] == 0) range[1] = landmarks.size();
    else range[1] = range[0] + range[1];

    cutted_landmarks.resize(range[1]-range[0]);

    for (int i = range[0], j = 0; i < range[1]; i+=2, j+=2) {
        cutted_landmarks[j] = float(landmarks[i] - cut_rect.x) / cut_rect.width;
        cutted_landmarks[j+1] = float(landmarks[i+1] - cut_rect.y) / cut_rect.height;
    }
}

void CartoonConstructor::renormalize(std::vector<float> &landmarks)
{
    cv::Rect2f bbox;
    CartoonConstructor::landmarks_to_rect(landmarks, bbox);
    float side_size = std::max(bbox.width, bbox.height);
    float dx = (side_size-bbox.width)*0.5;
    float dy = (side_size-bbox.height)*0.5;
    bbox.width = side_size;
    bbox.height = side_size;
    for (uint i = 0; i < landmarks.size(); i+=2) {
        landmarks[i] = (landmarks[i]-(bbox.x-dx))/bbox.width;
        landmarks[i+1] = (landmarks[i+1]-(bbox.y-dy))/bbox.height;
    }
}
