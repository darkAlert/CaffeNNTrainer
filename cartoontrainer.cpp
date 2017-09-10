#include "cartoontrainer.h"
#include "helper.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist
#include <opencv2/highgui.hpp>             // imshow, waitKey
#include <random>

#include "caffe/net.hpp"
#include <cuda.h>

#include <QDebug>
#include <QTime>
#include <QFile>
#include <QFileInfo>

using namespace caffe;


CartoonTrainer::CartoonTrainer()
{
    solver = nullptr;
    running = false;
    prefetch_worker_is_running = false;
    train_worker_is_running = false;
    test_net = nullptr;
    test_data.first = nullptr;
    test_data.second = nullptr;
}

CartoonTrainer::~CartoonTrainer()
{
    stop();
}

void CartoonTrainer::prefetch_batches_by_worker(unsigned int batch_margin)
{
    prefetch_worker_is_running = true;

    //Batch params:
    const int batch_size = solver->net()->blob_by_name("data")->shape(0);
    const int channels = solver->net()->blob_by_name("data")->shape(1);
    const int sample_size = channels;
    const int label_size = solver->net()->blob_by_name("label")->shape(1);

    struct Distortions {
        const bool mirror = false;

        const bool deform = false;
        const float max_deform_ratio = 0.5;
    } DISTORTS;

    //Distributions:
    boost::random::uniform_real_distribution<float> die_one(0.0, 1.0);
    boost::random::uniform_real_distribution<float> die_deform(1.0-DISTORTS.max_deform_ratio, 1.0+DISTORTS.max_deform_ratio);
    boost::random::uniform_int_distribution<int> die_id(0, cartoon_constructor_train->size()-1);
    rng.seed(std::time(nullptr));

    while (running)
    {
        mutex_prefetch_data.lock();
        if (prefetched_data.size() < batch_margin)
        {
            mutex_prefetch_data.unlock();

            //Main triplet net:
            std::shared_ptr<float> data_batch;
            std::shared_ptr<float> labels_batch;
            data_batch.reset(new float[batch_size*sample_size]);
            labels_batch.reset(new float[batch_size*label_size]);

//            /// Pick samples ///
//            for (int i = 0; i < batch_size; ++i)
//            {
//                const std::vector<float> raw_label = raw_train_samples[i].second;
//                const std::vector<float> raw_sample = raw_train_samples[i].first;
//                std::vector<float> sample(raw_sample.begin(), raw_sample.end());
//                std::vector<float> label(raw_label.begin(), raw_label.end());


//                //Set sample and label to the net:
//                memcpy(&(data_batch.get()[i*sample_size]), sample.data(), sizeof(float)*(sample_size));
//                memcpy(&(labels_batch.get()[i*label_size]), label.data(), sizeof(float)*(label_size));
//            }
            /// Pick samples ///
            for (int i = 0; i < batch_size; ++i)
            {
                //Get id (randomly):
                uint scaffold_id = die_id(rng);
                std::vector<uint> component_ids(FaceParts::size);
                component_ids[FaceParts::eye_l] = die_id(rng);
                component_ids[FaceParts::eye_r] = component_ids[FaceParts::eye_l];
                component_ids[FaceParts::brow_l] = die_id(rng);
                component_ids[FaceParts::brow_r] = component_ids[FaceParts::brow_l];
                component_ids[FaceParts::nose] = die_id(rng);
                component_ids[FaceParts::mouth] = die_id(rng);
                component_ids[FaceParts::contour] = die_id(rng);
//                component_ids[FaceParts::eye_l] = scaffold_id;
//                component_ids[FaceParts::eye_r] = scaffold_id;
//                component_ids[FaceParts::brow_l] = scaffold_id;
//                component_ids[FaceParts::brow_r] = scaffold_id;
//                component_ids[FaceParts::nose] = scaffold_id;
//                component_ids[FaceParts::mouth] = scaffold_id;
//                component_ids[FaceParts::contour] = scaffold_id;

                //Generate landmarks:
                std::vector<float> sample;
                std::vector<float> label;
                cartoon_constructor_train->generate(scaffold_id, component_ids, sample, label);
                assert(sample.size() == label.size());

                //Deformation:
                if (DISTORTS.deform){
                    float deform_ratio = die_deform(rng);
                    uint deform_axis = static_cast<uint>(floor(die_one(rng) + 0.5));
                    for (uint i = deform_axis; i < sample.size(); i+=2) {
                        sample[i] = sample[i]*deform_ratio;
                        label[i] = label[i]*deform_ratio;
                    }
                    CartoonConstructor::renormalize(sample);
                    CartoonConstructor::renormalize(label);
                }

                //Mirror:
                if (DISTORTS.mirror && die_one(rng) < 0.5)
                {
                    //Reorder sample landmarks:
                    for (unsigned int j = 0; j < sample.size(); j+=2) {
                        sample[j] = 1.0 - sample[j];
                    }
                    CartoonConstructor::swap_landmarks(sample, 0,7, 9,16);       // face contour
                    CartoonConstructor::swap_landmarks(sample, 17,21, 22,26);    // eyebrows
                    CartoonConstructor::swap_landmarks(sample, 31,32, 34,35);    // nose
                    CartoonConstructor::swap_landmarks(sample, 36,39, 42,45);    // eyes top
                    CartoonConstructor::swap_landmarks(sample, 40,41, 46,47);    // eyes bottom
                    CartoonConstructor::swap_landmarks(sample, 48,50, 52,54);    // mouth external top
                    CartoonConstructor::swap_landmarks(sample, 55,56, 58,59);    // mouth external bottom
                    CartoonConstructor::swap_landmarks(sample, 60,61, 63,64);    // mouth internal top
                    CartoonConstructor::swap_landmarks(sample, 65,65, 67,67);    // mouth internal bottom

                    //Reorder label landmarks:
                    for (unsigned int j = 0; j < label.size(); j+=2) {
                        label[j] = 1.0 - label[j];
                    }
                    CartoonConstructor::swap_landmarks(label, 0,7, 9,16);       // face contour
                    CartoonConstructor::swap_landmarks(label, 17,21, 22,26);    // eyebrows
                    CartoonConstructor::swap_landmarks(label, 31,32, 34,35);    // nose
                    CartoonConstructor::swap_landmarks(label, 36,39, 42,45);    // eyes top
                    CartoonConstructor::swap_landmarks(label, 40,41, 46,47);    // eyes bottom
                    CartoonConstructor::swap_landmarks(label, 48,50, 52,54);    // mouth external top
                    CartoonConstructor::swap_landmarks(label, 55,56, 58,59);    // mouth external bottom
                    CartoonConstructor::swap_landmarks(label, 60,61, 63,64);    // mouth internal top
                    CartoonConstructor::swap_landmarks(label, 65,65, 67,67);    // mouth internal bottom
                }

                //Set sample and label to the net:
                memcpy(&(data_batch.get()[i*sample_size]), sample.data(), sizeof(float)*(sample_size));
                memcpy(&(labels_batch.get()[i*label_size]), label.data(), sizeof(float)*(label_size));

/*
                /// DEBUG ///
                cv::Mat result_img = cv::Mat(500, 1000, CV_8UC3);
                {
                    cv::Mat canvas = cv::Mat::zeros(500,500, CV_8UC3);
                    canvas = cv::Scalar(0,0,0);
                    int line_size = 3;
                    for (unsigned int i = 0; i < sample.size(); i+=2) {
                        cv::Point2i point(floor(sample[i]*canvas.cols+0.5),
                                          floor(sample[i+1]*canvas.rows+0.5));
                        cv::Point2i pt1(point.x-line_size, point.y);
                        cv::Point2i pt2(point.x+line_size, point.y);
                        cv::Point2i pt3(point.x, point.y-line_size);
                        cv::Point2i pt4(point.x, point.y+line_size);
                        cv::line(canvas, pt1, pt2, cv::Scalar(0,255,0,0));
                        cv::line(canvas, pt3, pt4, cv::Scalar(0,255,0,0));
                    }
                    cv::Mat ref = result_img(cv::Rect(0,0,canvas.cols,canvas.rows));
                    canvas.copyTo(ref);
                }
                {
                    cv::Mat canvas = cv::Mat::zeros(500,500, CV_8UC3);
                    canvas = cv::Scalar(0,0,0);
                    int line_size = 3;
                    for (unsigned int i = 0; i < label.size(); i+=2) {
                        cv::Point2i point(floor(label[i]*canvas.cols+0.5),
                                          floor(label[i+1]*canvas.rows+0.5));
                        cv::Point2i pt1(point.x-line_size, point.y);
                        cv::Point2i pt2(point.x+line_size, point.y);
                        cv::Point2i pt3(point.x, point.y-line_size);
                        cv::Point2i pt4(point.x, point.y+line_size);
                        cv::line(canvas, pt1, pt2, cv::Scalar(0,255,0,0));
                        cv::line(canvas, pt3, pt4, cv::Scalar(0,255,0,0));
                    }
                    cv::Mat ref = result_img(cv::Rect(500,0,canvas.cols,canvas.rows));
                    canvas.copyTo(ref);
                }
                QString name = QString("/home/darkalert/Desktop/test/%1.png").arg(i);
                cv::imwrite(name.toStdString(), result_img);
//                qDebug() << "i=" << i << scaffold_id;
                ///////////
*/
            }


            //Push a prepared data batch to the prefetched_data:
            mutex_prefetch_data.lock();
            prefetched_data.push_back(std::make_pair(data_batch,labels_batch));
            mutex_prefetch_data.unlock();
        }
        else {
            mutex_prefetch_data.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(25000));
        }
    }

    prefetch_worker_is_running = false;
}

void CartoonTrainer::prepare_test_data()
{
    if (test_net == nullptr) return;

    //Data params:
    const int channels = test_net->blob_by_name("data")->shape(1);
    const int sample_size = channels;
    const int label_size = test_net->blob_by_name("label")->shape(1);
    actual_num_test_samples = std::pow(samples_per_test, 5+1);  //5 - components, +1 - scaffold

    //Allocate memory:
    test_data.first.reset(new float[actual_num_test_samples*sample_size]);
    test_data.second.reset(new float[actual_num_test_samples*label_size]);

    //Prepare data (generate all combinations):
    uint index = 0;
    for (uint scaffold_id = 0; scaffold_id < samples_per_test; ++scaffold_id) {
        for (uint eye_id = 0; eye_id < samples_per_test; ++eye_id) {
            for (uint brow_id = 0; brow_id < samples_per_test; ++brow_id) {
                for (uint mouth_id = 0; mouth_id < samples_per_test; ++mouth_id) {
                    for (uint nose_id = 0; nose_id < samples_per_test; ++nose_id) {
                        for (uint contour_id = 0; contour_id < samples_per_test; ++contour_id)
                        {
                            //Generate:
                            std::vector<uint> component_ids(FaceParts::size);
                            component_ids[FaceParts::eye_l] = eye_id;
                            component_ids[FaceParts::eye_r] = eye_id;
                            component_ids[FaceParts::brow_l] = brow_id;
                            component_ids[FaceParts::brow_r] = brow_id;
                            component_ids[FaceParts::nose] = nose_id;
                            component_ids[FaceParts::mouth] = mouth_id;
                            component_ids[FaceParts::contour] = contour_id;

                            std::vector<float> sample;
                            std::vector<float> label;
                            cartoon_constructor_test->generate(scaffold_id, component_ids, sample, label);
                            assert(sample.size() == label.size());

                            //Set data and label to the net:
                            memcpy(&(test_data.first.get())[index*sample_size], sample.data(), sizeof(float)*sample_size);
                            memcpy(&(test_data.second.get()[index*label_size]), label.data(), sizeof(float)*label_size);
                            ++index;
                        }
                    }
                }
            }
        }
    }
    assert(actual_num_test_samples == index);
    qDebug() << "CartoonTrainer: Total test samplex:" << actual_num_test_samples;
}

void CartoonTrainer::set_data_batch()
{
    auto net = solver->net();
    boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    const int sample_size = data_blob->shape(1);
    const int label_size = label_blob->shape(1);
    float* data = data_blob->mutable_cpu_data();
    float* labels = label_blob->mutable_cpu_data();
    bool waiting_flag = false;

    bool flag;
    do {
        //Check an availability of data:
        mutex_prefetch_data.lock();
        flag = prefetched_data.empty();
        mutex_prefetch_data.unlock();
        if (flag) {
            if (waiting_flag == false) {
                waiting_flag = true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(25000));
            continue;
        }
        //Take a data batch:
        mutex_prefetch_data.lock();
        auto data_batch = prefetched_data.front();
        prefetched_data.pop_front();
        mutex_prefetch_data.unlock();

        //Set the data and labels:
        memcpy(data, data_batch.first.get(), sizeof(float)*sample_size*num);
        memcpy(labels, data_batch.second.get(), sizeof(float)*label_size*num);
        flag = false;

    } while(flag);
}

void CartoonTrainer::set_test_data_batch(int sample_index)
{
    assert(test_net != nullptr);
    boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    const int sample_size = data_blob->shape(1);
    const int label_size = label_blob->shape(1);
    float* data = data_blob->mutable_cpu_data();
    float* labels = label_blob->mutable_cpu_data();

    //Set the data and labels:
    auto rest = actual_num_test_samples-sample_index >= (unsigned int)num ? (unsigned int)num : actual_num_test_samples-sample_index;
    memcpy(data, &(test_data.first.get()[sample_index*sample_size]), sizeof(float)*sample_size*rest);
    memcpy(labels, &(test_data.second.get()[sample_index*label_size]), sizeof(float)*label_size*rest);
}

bool CartoonTrainer::read_raw_samples(const std::string &path_to_csv)
{
    //Open csv containing dirs:
    std::ifstream csv_file(path_to_csv.c_str(), std::ifstream::in);
    if (!csv_file) {
        qDebug() << "CartoonTrainer: No valid input file was given, please check the given filename.";
        return false;
    }
    QString path_to_dir = QFileInfo(QFile(QString::fromStdString(path_to_csv))).absolutePath() + "/";

    //Allocate memory:
    raw_train_samples.clear();
    raw_test_samples.clear();
    raw_test_samples.reserve(samples_per_test);
    raw_train_samples.reserve(std::count(std::istreambuf_iterator<char>(csv_file), std::istreambuf_iterator<char>(), '\n') - samples_per_test);
    csv_file.clear();
    csv_file.seekg(0, std::ios::beg);  //back to the beginning

    //Read samples and labels by lines:
    std::string line;
    unsigned int count = 0;
    while (std::getline(csv_file, line))
    {
        //Get file names:
        size_t prev_pos = 0, pos = 0;
        pos = line.find(";", prev_pos);
        std::string path_to_sample = path_to_dir.toStdString() + line.substr(prev_pos,pos-prev_pos);
        std::string path_to_label = path_to_dir.toStdString() + line.substr(pos+1);

        //Open and parse sample csv:
        std::ifstream sample_file(path_to_sample.c_str(), std::ifstream::in);
        if (!sample_file) {
            qDebug() << QString::fromStdString(path_to_sample) << "missing.";
            continue;
        }
        std::vector<float> sample;
        std::string line_s;
        std::getline(sample_file, line_s);
        prev_pos = 0, pos = 0;
        while ((pos = line_s.find(";", prev_pos)) != std::string::npos) {
            sample.push_back(std::stof(line_s.substr(prev_pos,pos-prev_pos)));
            prev_pos = pos+1;
        }

        //Open and parse label csv:
        std::ifstream label_file(path_to_label.c_str(), std::ifstream::in);
        if (!label_file) {
            qDebug() << QString::fromStdString(path_to_label) << "missing.";
            continue;
        }
        std::vector<float> label;
        std::string line_l;
        std::getline(label_file, line_l);
        prev_pos = 0, pos = 0;
        while ((pos = line_l.find(";", prev_pos)) != std::string::npos) {
            label.push_back(std::stof(line_l.substr(prev_pos,pos-prev_pos)));
            prev_pos = pos+1;
        }

        /// HACK 68 as 69 ///
        label.pop_back();
        label.pop_back();
        assert(label.size() == 68*2);


        //Push sample and label:
        if (count < samples_per_test) {
            raw_test_samples.push_back(std::make_pair(sample, label));
        }
        else {
            raw_train_samples.push_back(std::make_pair(sample, label));
        }
        ++count;
    }
    assert(raw_test_samples.size() == samples_per_test);

    //Init cartoon constructor:
    cartoon_constructor_train = std::make_shared<CartoonConstructor>(raw_train_samples);
    cartoon_constructor_test = std::make_shared<CartoonConstructor>(raw_test_samples);

    qDebug() << "CartoonTrainer: Sample have been loaded. Train sampels:" << raw_train_samples.size() << ", test samples:" << raw_test_samples.size();
    qDebug() << "CartoonTrainer: Cartoon constructor have been initialized. Train samples:" << cartoon_constructor_train->size() <<
                ", test:" << cartoon_constructor_test->size();

    return true;
}

void CartoonTrainer::train_by_worker()
{
    train_worker_is_running = true;

    //Solver's params:
    auto param = solver->param();
    const int32_t test_interval = param.test_interval();

    //Set mode:
    if (param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
        Caffe::SetDevice(0);
        Caffe::set_solver_count(1);
        Caffe::set_mode(Caffe::GPU);
    }
    else {
        Caffe::set_mode(Caffe::CPU);
    }

    //Prepare test data:
    prepare_test_data();

    while(running)
    {
        /// Test phase ///
        if (solver->iter() % test_interval == 0)
        {
            const float acc_threshold = 0.010;//0.010;   // in pixels
            Caffe::set_mode(Caffe::GPU);
            const int num = test_net->blob_by_name("data")->shape(0);
            boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
            boost::shared_ptr<Blob<float> > output_blob = test_net->blob_by_name("output");
            boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
            const int label_size = test_net->blob_by_name("output")->shape(1);
            float test_loss = 0.0;
            float test_points_acc = 0.0;
            float test_x_acc = 0.0, test_y_acc = 0.0;
            float test_contour_acc = 0.0, test_nose_acc = 0.0, test_mouth_acc = 0.0, test_eyes_acc = 0.0, test_eyebrows_acc = 0.0;
            float iters_count = 0;

            for (unsigned int si = 0; si < actual_num_test_samples; si += num) {
                set_test_data_batch(si);
                float loss = 0.0;
                test_net->Forward(&loss);   //forward
                test_loss += loss;

                //Calculate the output difference:
                const float* target = label_blob->cpu_data();
                const float* actual = output_blob->cpu_data();
                float points_acc = 0.0;
                float x_acc = 0.0, y_acc = 0.0;
                float contour_acc = 0.0, nose_acc = 0.0, mouth_acc = 0.0, eyes_acc = 0.0, eyebrows_acc = 0.0;

                for (int s = 0; s < num; ++s)
                {
                    for (int j = 0; j < label_size; j+=2)
                    {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        x_acc += int(sqrt(dif_x) < acc_threshold);
                        y_acc += int(sqrt(dif_y) < acc_threshold);
                        points_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Contour:
                    for (uint j = Landmarks68p::contour_index*2; j < (Landmarks68p::contour_index+Landmarks68p::contour_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        contour_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Eyebrows left:
                    for (uint j = Landmarks68p::eyebrow_left_index*2; j < (Landmarks68p::eyebrow_left_index+Landmarks68p::eyebrow_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        eyebrows_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Eyebrows right:
                    for (uint j = Landmarks68p::eyebrow_right_index*2; j < (Landmarks68p::eyebrow_right_index+Landmarks68p::eyebrow_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        eyebrows_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Nose:
                    for (uint j = Landmarks68p::nose_index*2; j < (Landmarks68p::nose_index+Landmarks68p::nose_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        nose_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Eye left:
                    for (uint j = Landmarks68p::eye_left_index*2; j < (Landmarks68p::eye_left_index+Landmarks68p::eye_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        eyes_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Eye right:
                    for (uint j = Landmarks68p::eye_right_index*2; j < (Landmarks68p::eye_right_index+Landmarks68p::eye_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        eyes_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    //Mouth:
                    for (uint j = Landmarks68p::mouth_index*2; j < (Landmarks68p::mouth_index+Landmarks68p::mouth_count)*2; j+=2) {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        mouth_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                }

                /*
                //DEBUG:
                {
                    const float* src_landmarks = data_blob->cpu_data();
                    const float* dst_landmarks_target = label_blob->cpu_data();
                    const float* dst_landmarks_actual = output_blob->cpu_data();
                    const int line_size = 5;
                    cv::Size out_size(550, 250);

                    for (unsigned int j = 0; j < num; ++j)
                    {
                        cv::Mat out_img = cv::Mat(out_size.height, out_size.width, CV_8UC3);
                        out_img = cv::Scalar(0,0,0);

                        //Draw src_landmarks:
                        for (int k = 0; k < label_size; k+=2) {
                            float x = src_landmarks[label_size*j + k];
                            float y = src_landmarks[label_size*j + k+1];
                            cv::Point2i point(int(floor(x*250+0.5)), int(floor(y*out_size.height+0.5)));
                            cv::Point2i pt1(point.x-line_size, point.y);
                            cv::Point2i pt2(point.x+line_size, point.y);
                            cv::Point2i pt3(point.x, point.y-line_size);
                            cv::Point2i pt4(point.x, point.y+line_size);
                            cv::line(out_img, pt1, pt2, cv::Scalar(0,0,255));
                            cv::line(out_img, pt3, pt4, cv::Scalar(0,0,255));
                        }
                        //Draw dst_landmarks_target:
                        for (int k = 0; k < label_size; k+=2) {
                            float x = dst_landmarks_target[label_size*j + k];
                            float y = dst_landmarks_target[label_size*j + k+1];
                            cv::Point2i point(int(floor(x*250+300+0.5)), int(floor(y*out_size.height+0.5)));
                            cv::Point2i pt1(point.x-line_size, point.y);
                            cv::Point2i pt2(point.x+line_size, point.y);
                            cv::Point2i pt3(point.x, point.y-line_size);
                            cv::Point2i pt4(point.x, point.y+line_size);
                            cv::line(out_img, pt1, pt2, cv::Scalar(255,0,0));
                            cv::line(out_img, pt3, pt4, cv::Scalar(255,0,0));
                        }
                        //Draw dst_landmarks_actual:
                        for (int k = 0; k < label_size; k+=2) {
                            float x = dst_landmarks_actual[label_size*j + k];
                            float y = dst_landmarks_actual[label_size*j + k+1];
                            cv::Point2i point(int(floor(x*250+300+0.5)), int(floor(y*out_size.height+0.5)));
                            cv::Point2i pt1(point.x-line_size, point.y);
                            cv::Point2i pt2(point.x+line_size, point.y);
                            cv::Point2i pt3(point.x, point.y-line_size);
                            cv::Point2i pt4(point.x, point.y+line_size);
                            cv::line(out_img, pt1, pt2, cv::Scalar(0,255,0));
                            cv::line(out_img, pt3, pt4, cv::Scalar(0,255,0));
                        }

                        QString img_name = QString("/home/darkalert/Desktop/test/%1.png").arg(si+j);
                        cv::imwrite(img_name.toStdString(), out_img);
                    }
                }
                */


                test_x_acc += x_acc/float(label_size*0.5*num);
                test_y_acc += y_acc/float(label_size*0.5*num);
                test_points_acc += points_acc/float(label_size*0.5*num);
                test_contour_acc += contour_acc/float(Landmarks68p::contour_count*num);
                test_eyebrows_acc += eyebrows_acc/float(Landmarks68p::eyebrow_count*2*num);
                test_nose_acc += nose_acc/float(Landmarks68p::nose_count*num);
                test_eyes_acc += eyes_acc/float(Landmarks68p::eye_count*2*num);
                test_mouth_acc += mouth_acc/float(Landmarks68p::mouth_count*num);

                ++iters_count;
            }

            qDebug() << "Test phase, iteration" << solver->iter();
            test_loss /= iters_count;
            test_points_acc /= iters_count;
            test_x_acc /= iters_count;
            test_y_acc /= iters_count;
            float test_xy_acc = (test_x_acc + test_y_acc)*0.5;
            test_contour_acc /= iters_count;
            test_eyebrows_acc /= iters_count;
            test_nose_acc /= iters_count;
            test_eyes_acc /= iters_count;
            test_mouth_acc /= iters_count;
            qDebug() << "   Test net: loss =" << test_loss <<  ", points_acc =" << test_points_acc << " (error:" << acc_threshold*100.0 << "%)"
                     << ", x_acc =" << test_x_acc << ", y_acc =" << test_y_acc << ", xy_acc =" << test_xy_acc;
            qDebug() << "   Acc components: contour =" << test_contour_acc <<  ", eyebrows =" << test_eyebrows_acc <<  ", nose =" << test_nose_acc
                        <<  ", eyes =" << test_eyes_acc <<  ", mouth =" << test_mouth_acc;
        }

        /// Treain phase ///
        Caffe::set_mode(Caffe::GPU);
        set_data_batch();
        solver->Step(1);
    }

    train_worker_is_running = false;
}

void CartoonTrainer::train(const std::string &path_to_solver)
{
    if (running == true) {
        qDebug() << "CartoonTrainer: Training is already running.";
        return;
    }

    if (raw_train_samples.empty()) {
        qDebug() << "CartoonTrainer: Training data are empty!";
    }

    //Load solver:
    Caffe::set_mode(Caffe::GPU);
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(path_to_solver, &solver_param);
    solver_param.set_test_initialization(false);
    solver = std::make_shared<SGDSolver<float> >(solver_param);

    //Load test net:
    caffe::NetParameter test_net_param;
    caffe::ReadNetParamsFromTextFileOrDie(solver_param.net(), &test_net_param);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, test_net_batch_size);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(1)->set_dim(0, test_net_batch_size);
    test_net_param.mutable_state()->set_phase(Phase::TEST);
    test_net = std::make_shared<Net<float> >(test_net_param);
    solver->net().get()->ShareTrainedLayersWith(test_net.get());

    //Load weights:
    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/snap/p68_deform_v2_iter_50000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/snap/p68_deform_v1_iter_80000.caffemodel");


    //Log:
    qDebug() << "CartoonTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "CartoonTrainer: GPU mode has been set." : "CartoonTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&CartoonTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&CartoonTrainer::train_by_worker, this);
    train_worker.detach();
}

void CartoonTrainer::restore(const std::string &path_to_solver, const std::string &path_to_solverstate)
{
    if (running == true) {
        qDebug() << "CartoonTrainer: Training is already running.";
        return;
    }

    if (raw_train_samples.empty()) {
        qDebug() << "CartoonTrainer: Training data are empty!";
    }

    //Load solver:
    Caffe::set_mode(Caffe::GPU);
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(path_to_solver, &solver_param);
    solver_param.set_test_initialization(false);
    solver = std::make_shared<SGDSolver<float> >(solver_param);

    //Load test net:
    caffe::NetParameter test_net_param;
    caffe::ReadNetParamsFromTextFileOrDie(solver_param.net(), &test_net_param);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, test_net_batch_size);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(1)->set_dim(0, test_net_batch_size);
    test_net_param.mutable_state()->set_phase(Phase::TEST);
    test_net = std::make_shared<Net<float> >(test_net_param);
    solver->net().get()->ShareTrainedLayersWith(test_net.get());

    //Restore state:
    solver->Restore(path_to_solverstate.c_str());

    //Log:
    qDebug() << "CartoonTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << "CartoonTrainer: Solver's state has been restored from" << QString::fromStdString(path_to_solverstate);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "CartoonTrainer: GPU mode has been set." : "CartoonTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&CartoonTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&CartoonTrainer::train_by_worker, this);
    train_worker.detach();
}

void CartoonTrainer::stop()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        helper::little_sleep(std::chrono::microseconds(100));
    }

    //Snapshot:
    if (solver != nullptr) {
        solver->Snapshot();
    }

    solver = nullptr;
    test_net = nullptr;
    test_data.first = nullptr;
    test_data.second = nullptr;
}

void CartoonTrainer::pause()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        helper::little_sleep(std::chrono::microseconds(100));
    }

    qDebug() << "ClassifierTrainer: Training is paused. Current iter:" << solver->iter();
}

void CartoonTrainer::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&CartoonTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&CartoonTrainer::train_by_worker, this);
    train_worker.detach();

    qDebug() << "CartoonTrainer: Training is resumed.";
}
