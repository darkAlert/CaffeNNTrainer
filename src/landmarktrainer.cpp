#include "landmarktrainer.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include <QFileInfo>
#include <QDebug>
#include <QTime>
#include <QDir>

#include "caffe/net.hpp"
#include <cuda.h>
#include <opencv2/cudawarping.hpp>

using namespace caffe;

#define LABEL_LANDMARKS_68P 0
#define LABEL_LANDMARKS_51P 1  //without contour points
#define LABEL_LANDMARKS_17P 2  //with contour points
#define LABEL_RECT_2P 3

const int LABEL_TYPE = LABEL_LANDMARKS_68P;

#define USE_CALSSIFIER 1

#define CONTOUR_INDEX 0
#define CONTOUR_COUNT 17
#define EYEBROWS_INDEX 17
#define EYEBROWS_COUNT 10
#define NOSE_INDEX 27
#define NOSE_COUNT 9
#define EYES_INDEX 36
#define EYES_COUNT 12
#define MOUTH_INDEX 48
#define MOUTH_COUNT 20


LandmarkTrainer::LandmarkTrainer()
{
    data_provider = nullptr;
    solver = nullptr;
    running = false;
    prefetch_worker_is_running = false;
    train_worker_is_running = false;
    test_net = nullptr;
    test_data.first = nullptr;
    test_data.second = nullptr;
    num_test_data = 0;
}

LandmarkTrainer::~LandmarkTrainer()
{
    stop();
}

void LandmarkTrainer::prefetch_batches_by_worker(unsigned int batch_margin)
{
    prefetch_worker_is_running = true;

    //Batch params:
    const int batch_size = solver->net()->blob_by_name("data")->shape(0);
    const int channels = solver->net()->blob_by_name("data")->shape(1);
    const int height = solver->net()->blob_by_name("data")->shape(2);
    const int width = solver->net()->blob_by_name("data")->shape(3);
    const int sample_size = channels*height*width;
    const int label_size = solver->net()->blob_by_name("label")->shape(1) * solver->net()->blob_by_name("label")->shape(2) * solver->net()->blob_by_name("label")->shape(3);

    struct Distortions {
        const bool crop = true;
        const cv::Size2i actual_sample_size = cv::Size2i(200,200);
//        const cv::Size2i actual_sample_size = cv::Size2i(240,240);
        const cv::Size2i origin_sample_size = cv::Size2i(160,160);

        const bool mirror = true;

        const bool scale = true;
        const float max_scale = std::max(actual_sample_size.width-origin_sample_size.width,
                                         actual_sample_size.height-origin_sample_size.height);
        const int scale_r1 = max_scale/2;
        const int scale_r2 = std::max(actual_sample_size.width,actual_sample_size.height)-scale_r1;

        const bool rotate = true;
        const float max_rotate_angle = 15.0;

        const bool blackbox = true;
        const float max_blackbox_size = 0.5;

        const bool deform = true;
        const float max_deform_ratio = 1.20;   //must be greater 1.0 !!!!

        const bool grayscale = true;
        const float grayscale_prob = 0.5;

        const bool resize = false;
        const float max_resize_ratio = 0.5;
    } DISTORTS;

    //Distributions:
    srand(std::time(nullptr));
    rng.seed(std::time(nullptr));
    boost::random::uniform_real_distribution<float> die_one(0.0, 1.0);
    boost::random::uniform_int_distribution<int> die_scale(0, DISTORTS.max_scale);
    boost::random::uniform_real_distribution<float> die_rotate(-DISTORTS.max_rotate_angle, DISTORTS.max_rotate_angle);
    boost::random::uniform_real_distribution<float> die_deform(1.0, DISTORTS.max_deform_ratio);
    boost::random::uniform_real_distribution<float> die_blackbox(0.0, DISTORTS.max_blackbox_size);
    boost::random::uniform_real_distribution<float> die_resize(0.0, DISTORTS.max_resize_ratio);

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

            /// Pick samples ///
            for (int i = 0; i < batch_size; ++i)
            {
                std::vector<float> raw_label;
                auto raw_sample = data_provider->get_train_sample(raw_label);
                std::vector<float> label(raw_label.begin(), raw_label.end());
                cv::Mat sample_float;
                int type_i = raw_sample.channels() == 1 ? CV_8UC1 : CV_8UC3;
                int type_f = raw_sample.channels() == 1 ? CV_32FC1 : CV_32FC3;
                cv::Mat(raw_sample.rows, raw_sample.cols, type_i, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, type_f, 0.00390625);

                Deformation:
                if (DISTORTS.deform){
                    float deform_ratio = die_deform(rng);
                    uint deform_axis = static_cast<uint>(floor(die_one(rng) + 0.5));
                    if (deform_axis == 0) {
                        cv::resize(sample_float, sample_float, cv::Size(0,0), deform_ratio, 1.0, CV_INTER_LINEAR);
                    }
                    else {
                        cv::resize(sample_float, sample_float, cv::Size(0,0), 1.0, deform_ratio, CV_INTER_LINEAR);
                    }
//                    //Label:
//                    for (uint j = deform_axis; j < label.size(); j+=2) {
//                        label[j] = label[j]*deform_ratio;
//                    }
                }
                //Rotate:
                if (DISTORTS.rotate) {
                    float angle = die_rotate(rng);
                    auto rot_mat = cv::getRotationMatrix2D(cv::Point(floor(sample_float.cols*0.5+0.5), floor(sample_float.rows*0.5+0.5)), angle, 1.0);
                    if (gpu_mode == true) {
                        cv::cuda::GpuMat frame_gpu(sample_float);
                        cv::cuda::GpuMat frame_rotated_gpu;
                        cv::cuda::warpAffine(frame_gpu, frame_rotated_gpu, rot_mat, cv::Size(sample_float.cols, sample_float.rows));
                        frame_rotated_gpu.download(sample_float);
                    }
                    else {
                        cv::warpAffine(sample_float, sample_float, rot_mat, cv::Size(sample_float.cols, sample_float.rows));
                    }
                    //Rotate landmarks:
                    const cv::Point2f center(0.5, 0.5);
                    const float cos_a = std::cos(-angle*(CV_PI/180.0));
                    const float sin_a = std::sin(-angle*(CV_PI/180.0));
                    for (unsigned int j = 0; j < label.size(); j+=2) {
                        label[j] -= center.x;
                        label[j+1] -= center.y;
                        label[j] = cos_a*label[j] - sin_a*label[j+1];
                        label[j+1] = sin_a*label[j] +cos_a*label[j+1];
                        label[j] += center.x;
                        label[j+1] += center.y;
                    }
                }
                //Scale and crop:
                if (true) {
                    int scale = die_scale(rng);
                    cv::Rect2i crop_rect(0,0,
                                         DISTORTS.origin_sample_size.width+scale, DISTORTS.origin_sample_size.height+scale);
                    int offset = DISTORTS.scale_r2 - std::max(crop_rect.width, crop_rect.height);
                    offset = offset < 0 ? 0 : offset;
                    int max_step = std::max(DISTORTS.actual_sample_size.width, DISTORTS.actual_sample_size.height) -
                            std::max(crop_rect.width, crop_rect.height) - offset*2;
                    crop_rect.x = offset + rand()%(max_step+1);
                    crop_rect.y = crop_rect.x;
                    //Correct the landmarks:
                    for (unsigned int j = 0; j < label.size(); j+=2) {
                        label[j] = (label[j]*sample_float.cols - crop_rect.x)/crop_rect.width;
                        label[j+1] = (label[j+1]*sample_float.rows - crop_rect.y)/crop_rect.height;
                    }
                    sample_float = sample_float(crop_rect).clone();
                }
//                //Scale:
//                if (DISTORTS.scale) {
//                    float ratio = 1.1;// + die_scale(rng);
//                    cv::Mat scaled_sample;
//                    cv::resize(sample_float, scaled_sample, cv::Size(0,0), ratio, ratio, CV_INTER_CUBIC);
//                    cv::Size new_size(floor(sample_float.cols/ratio+0.5), floor(sample_float.rows/ratio+0.5));
//                    cv::Rect crop_rect(floor((scaled_sample.cols-new_size.width)*0.5 + 0.5), floor((scaled_sample.rows-new_size.height)*0.5 + 0.5),
//                                       new_size.width, new_size.height);
//                    /*cv::Rect crop_rect(floor((scaled_sample.cols-sample_float.cols)*0.5 + 0.5), floor((scaled_sample.rows-sample_float.rows)*0.5 + 0.5),
//                                       sample_float.cols, sample_float.rows)*/;
//                    sample_float = scaled_sample(crop_rect).clone();
//                }
//                //Crop:
//                if (DISTORTS.crop && (sample_float.cols != width || sample_float.rows != height)) {
//                    int dif_w = std::min(sample_float.cols,DISTORTS.max_crop_offset.width) - width;
//                    int dif_h = std::min(sample_float.rows,DISTORTS.max_crop_offset.height) - height;
//                    cv::Rect crop_rect(0,0,width,height);
//                    crop_rect.x = rand()%(dif_w+1) + std::max(int(floor((sample_float.cols-DISTORTS.max_crop_offset.width)*0.5+0.5)),0);
//                    crop_rect.y = rand()%(dif_h+1) + std::max(int(floor((sample_float.rows-DISTORTS.max_crop_offset.height)*0.5+0.5)),0);
//                    //Correct the landmarks:
//                    for (unsigned int j = 0; j < label.size(); j+=2) {
//                        label[j] = (label[j]*sample_float.cols - crop_rect.x)/crop_rect.width;
//                        label[j+1] = (label[j+1]*sample_float.rows - crop_rect.y)/crop_rect.height;
//                    }
//                    sample_float = sample_float(crop_rect).clone();
//                }
                //Mirror:
                if (DISTORTS.mirror && rand() < RAND_MAX/2) {
                    cv::flip(sample_float,sample_float,1);
                    //Mirror landmarks:
                    for (unsigned int j = 0; j < label.size(); j+=2) {
                        label[j] = 1.0 - label[j];
                    }
                    if (LABEL_TYPE == LABEL_LANDMARKS_68P) {
                        //Reorder landmarks:
                        swap_landmarks(label, 0,7, 9,16);       // face contour
                        swap_landmarks(label, 17,21, 22,26);    // eyebrows
                        swap_landmarks(label, 31,32, 34,35);    // nose
                        swap_landmarks(label, 36,39, 42,45);    // eyes top
                        swap_landmarks(label, 40,41, 46,47);    // eyes bottom
                        swap_landmarks(label, 48,50, 52,54);    // mouth external top
                        swap_landmarks(label, 55,56, 58,59);    // mouth external bottom
                        swap_landmarks(label, 60,61, 63,64);    // mouth internal top
                        swap_landmarks(label, 65,65, 67,67);    // mouth internal bottom
                    }
                    else if(LABEL_TYPE == LABEL_LANDMARKS_51P) {
                        //Reorder landmarks:
                        const int offset = 17;
                        swap_landmarks(label, 17-offset,21-offset, 22-offset,26-offset);    // eyebrows
                        swap_landmarks(label, 31-offset,32-offset, 34-offset,35-offset);    // nose
                        swap_landmarks(label, 36-offset,39-offset, 42-offset,45-offset);    // eyes top
                        swap_landmarks(label, 40-offset,41-offset, 46-offset,47-offset);    // eyes bottom
                        swap_landmarks(label, 48-offset,50-offset, 52-offset,54-offset);    // mouth external top
                        swap_landmarks(label, 55-offset,56-offset, 58-offset,59-offset);    // mouth external bottom
                        swap_landmarks(label, 60-offset,61-offset, 63-offset,64-offset);    // mouth internal top
                        swap_landmarks(label, 65-offset,65-offset, 67-offset,67-offset);    // mouth internal bottom
                    }
                    else if(LABEL_TYPE == LABEL_LANDMARKS_17P) {
                        //Reorder landmarks:
                        swap_landmarks(label, 0,7, 9,16);       // face contour
                    }
                    else if (LABEL_TYPE == LABEL_RECT_2P) {
                        //Reorder facial rect:
                        std::swap(label[0], label[2]);
    //                    std::swap(label[1], label[3]); //only x
                    }
                }
                if (DISTORTS.blackbox) {
                    cv::Rect2i rect;
                    rect.x = floor(die_one(rng)*sample_float.cols + 0.5);
                    rect.y = floor(die_one(rng)*sample_float.rows + 0.5);
                    rect.width = floor(die_blackbox(rng)*sample_float.cols + 0.5);
                    rect.height = floor(die_blackbox(rng)*sample_float.rows + 0.5);
                    if (rect.x + rect.width > sample_float.cols) {
                        rect.width = sample_float.cols - rect.x;
                    }
                    if (rect.y + rect.height > sample_float.rows) {
                        rect.height = sample_float.rows - rect.y;
                    }
                    cv::rectangle(sample_float, rect, cv::Scalar(0,0,0), CV_FILLED);
                }
                if (DISTORTS.grayscale && rand()/float(RAND_MAX) < DISTORTS.grayscale_prob) {
                    if (sample_float.channels() == 3) {
                        cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2GRAY);
                    }
                    cv::cvtColor(sample_float, sample_float, cv::COLOR_GRAY2BGR);
                }
                //Resize:
                if (DISTORTS.resize) {
                    float ratio = 1.0 - die_resize(rng);
                    cv::resize(sample_float,sample_float,cv::Size(0, 0), ratio, ratio, CV_INTER_AREA);
                    cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, CV_INTER_LINEAR);
                }
                //Resize to fit:
                if (sample_float.rows != height || sample_float.cols != width) {
                    auto inter = sample_float.cols > width ? CV_INTER_AREA : CV_INTER_LINEAR;
                    cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, inter);
                }

                //Normalize landmarks (only for classifier):
#if USE_CALSSIFIER == 1
                {
                    for (uint i = 0; i < label.size(); i+=2) {
                        label[i] = floor(label[i]*sample_float.cols+0.5);
                        label[i+1] = floor(label[i+1]*sample_float.rows+0.5);
                    }
                }
#endif

//                //DEBUG:
//                {
//                    static int counter = 0;
//                    cv::Mat temp = sample_float.clone();
//                    temp = temp/0.00390625;

//                    if (LABEL_TYPE == LABEL_LANDMARKS_68P || LABEL_TYPE == LABEL_LANDMARKS_51P || LABEL_TYPE == LABEL_LANDMARKS_17P) {
//                        int line_size = int(floor(sample_float.cols/100.0*2.5/2.0 + 0.5));
//                        //Draw landmarks:
//                        int color_step = floor(255.0/(label.size()/2.0)+0.5);
//                        for (unsigned int j = 0; j < label.size(); j+=2) {
//                            cv::Point2i point(int(floor(label[j]*sample_float.cols+0.5)), int(floor(label[j+1]*sample_float.rows+0.5)));
//                            cv::Point2i pt1(point.x-line_size, point.y);
//                            cv::Point2i pt2(point.x+line_size, point.y);
//                            cv::Point2i pt3(point.x, point.y-line_size);
//                            cv::Point2i pt4(point.x, point.y+line_size);
//                            cv::line(temp, pt1, pt2, cv::Scalar(j/2*color_step, 255-j/2*color_step,255));
//                            cv::line(temp, pt3, pt4, cv::Scalar(j/2*color_step, 255-j/2*color_step,255));
//                        }
//                    }
//                    else if (LABEL_TYPE == LABEL_RECT_2P) {
//                        cv::Point2i pt1(int(floor(label[0]*sample_float.cols+0.5)), int(floor(label[1]*sample_float.rows+0.5)));
//                        cv::Point2i pt3(int(floor(label[2]*sample_float.cols+0.5)), int(floor(label[3]*sample_float.rows+0.5)));
//                        cv::Point2i pt2(pt3.x, pt1.y);
//                        cv::Point2i pt4(pt1.x, pt3.y);
//                        cv::line(temp, pt1, pt2, cv::Scalar(255,0,0));
//                        cv::line(temp, pt2, pt3, cv::Scalar(0,255,0));
//                        cv::line(temp, pt3, pt4, cv::Scalar(0,0,255));
//                        cv::line(temp, pt4, pt1, cv::Scalar(255,0,255));
//                    }

//                    QString img_name = QString("/home/darkalert/Desktop/test/%1_%2.png").arg(counter).arg(i);
//                    cv::imwrite(img_name.toStdString(), temp);
//                }

                //Set data to the main triplet net:
                if (sample_float.channels() == 1) {
                    //Grayscale:
                    memcpy(&(data_batch.get()[i*sample_size]), sample_float.data, sizeof(float)*(sample_size));
                }
                else {
                    //Rgb:
                    cv::Mat channels[3];
                    cv::split(sample_float,channels);
                    int channel_size = sample_size/3;
                    memcpy(&(data_batch.get()[i*sample_size]), channels[0].data, sizeof(float)*(channel_size));
                    memcpy(&(data_batch.get()[i*sample_size+channel_size]), channels[1].data, sizeof(float)*(channel_size));
                    memcpy(&(data_batch.get()[i*sample_size+channel_size*2]), channels[2].data, sizeof(float)*(channel_size));
                }
                memcpy(&(labels_batch.get()[i*label_size]), label.data(), sizeof(float)*(label_size));
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

void LandmarkTrainer::prepare_test_data()
{
    if (test_net == nullptr) return;

    //Data params:
    const int channels =test_net->blob_by_name("data")->shape(1);
    const int height = test_net->blob_by_name("data")->shape(2);
    const int width = test_net->blob_by_name("data")->shape(3);
    const int sample_size = channels*height*width;
    const int label_size = test_net->blob_by_name("label")->shape(1) * test_net->blob_by_name("label")->shape(2) * test_net->blob_by_name("label")->shape(3);
    num_test_data = num_test_samples;//data_provider->total_test_samples();

    //Allocate memory:
    test_data.first.reset(new float[num_test_data*sample_size]);
    test_data.second.reset(new float[num_test_data*label_size]);

    //Prepare data:
    for (unsigned int i = 0; i < num_test_data; ++i)
    {
        std::vector<float> raw_label;
        auto raw_sample = data_provider->get_test_sample(i, raw_label);
        std::vector<float> label(raw_label.begin(), raw_label.end());
        cv::Mat sample_float;
        int type_i = raw_sample.channels() == 1 ? CV_8UC1 : CV_8UC3;
        int type_f = raw_sample.channels() == 1 ? CV_32FC1 : CV_32FC3;
        cv::Mat(raw_sample.rows, raw_sample.cols, type_i, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, type_f, 0.00390625);
        //Crop:
        if (sample_float.cols != width || sample_float.rows != height) {
            int dif_w = sample_float.cols - width;
            int dif_h = sample_float.rows - height;
            cv::Rect crop_rect(0,0,width,height);
            crop_rect.x = floor(dif_w*0.5 + 0.5);
            crop_rect.y = floor(dif_h*0.5 + 0.5);
            //Correct the landmarks:
            for (unsigned int j = 0; j < label.size(); j+=2) {
                label[j] = (label[j]*sample_float.cols - crop_rect.x)/crop_rect.width;
                label[j+1] = (label[j+1]*sample_float.rows - crop_rect.y)/crop_rect.height;
            }
            sample_float = sample_float(crop_rect).clone();
        }
        //Resize to fit:
        if (sample_float.rows != height || sample_float.cols != width) {
            auto inter = sample_float.cols > width ? CV_INTER_AREA : CV_INTER_LINEAR;
            cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, inter);
        }
        //Normalize landmarks (only for classifier):
#if USE_CALSSIFIER == 1
        {
            for (uint i = 0; i < label.size(); i+=2) {
                label[i] = floor(label[i]*sample_float.cols+0.5);
                label[i+1] = floor(label[i+1]*sample_float.rows+0.5);
            }
        }
#endif
        //Set data to the main triplet net:
        if (sample_float.channels() == 1) {
            //Grayscale:
            memcpy(&(test_data.first.get())[i*sample_size], sample_float.data, sizeof(float)*sample_size);
        }
        else {
            //Rgb:
            cv::Mat channels[3];
            cv::split(sample_float,channels);
            int channel_size = sample_size/3;
            memcpy(&(test_data.first.get()[i*sample_size]), channels[0].data, sizeof(float)*(channel_size));
            memcpy(&(test_data.first.get()[i*sample_size+channel_size]), channels[1].data, sizeof(float)*(channel_size));
            memcpy(&(test_data.first.get()[i*sample_size+channel_size*2]), channels[2].data, sizeof(float)*(channel_size));
        }
        memcpy(&(test_data.second.get()[i*label_size]), label.data(), sizeof(float)*label_size);
    }

//    data_provider->clear_test_batch();
}

void LandmarkTrainer::set_data_batch()
{
    auto net = solver->net();
    boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    const int sample_size = data_blob->shape(3) * data_blob->shape(2) * data_blob->shape(1);
    const int label_size = label_blob->shape(3) * label_blob->shape(2) * label_blob->shape(1);
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
//                qDebug() << "Waiting for data...";
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

void LandmarkTrainer::set_test_data_batch(int sample_index)
{
    assert(test_net != nullptr);
    boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    assert(num == test_net_batch_size);
    const int sample_size = data_blob->shape(3) * data_blob->shape(2) * data_blob->shape(1);
    const int label_size = label_blob->shape(3) * label_blob->shape(2) * label_blob->shape(1);
    float* data = data_blob->mutable_cpu_data();
    float* labels = label_blob->mutable_cpu_data();

    //Set the data and labels:
    auto rest = num_test_samples-sample_index >= num ? num : num_test_samples-sample_index;
    memcpy(data, &(test_data.first.get()[sample_index*sample_size]), sizeof(float)*sample_size*rest);
    memcpy(labels, &(test_data.second.get()[sample_index*label_size]), sizeof(float)*label_size*rest);

//    cv::Mat out_img = sample_float.clone()/0.00390625;
//    QString img_name = QString("/home/darkalert/Desktop/test/%1.png").arg(i);
//    cv::imwrite(img_name.toStdString(), out_img);
}

void LandmarkTrainer::openTrainData(const std::string &path_to_train)
{
    //Open data:
    if (data_provider == nullptr) {
        data_provider = std::make_shared<DataProviderAttributes>(train_batch_size, data_provider_worker_count, imread_type);
    }
    if (data_provider->open(path_to_train, num_test_samples) == false) {
        qDebug() << "ClassifierTrainer: Unknown error while data loading. Training has been stopped.";
        return;
    }
}

void LandmarkTrainer::train_by_worker()
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

    //Logger:
    std::ofstream train_log(param.snapshot_prefix() + "train_log.txt", std::ios::app);

    while(running)
    {
        /// Test phase ///
        if (solver->iter() % test_interval == 0)
        {
            const float acc_threshold = 0.05*160;   // in pixels
            Caffe::set_mode(Caffe::GPU);
            const int num = test_net->blob_by_name("data")->shape(0);
            boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
            boost::shared_ptr<Blob<float> > landmarks_blob = test_net->blob_by_name("output");
            boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
//            const int label_size = test_net->blob_by_name("landmarks")->shape(1);
            const int label_size = landmarks_blob->shape(1)*landmarks_blob->shape(2)*landmarks_blob->shape(3);

            float test_loss = 0.0;
            float test_points_acc = 0.0;
            float test_x_acc = 0.0, test_y_acc = 0.0;
            float test_contour_acc = 0.0, test_nose_acc = 0.0, test_mouth_acc = 0.0, test_eyes_acc = 0.0, test_eyebrows_acc = 0.0;


            for (unsigned int i = 0; i < num_test_data; i += num) {
                set_test_data_batch(i);
                float loss = 0.0;
//                const std::vector<Blob<float>*>& result = test_net->Forward(&loss);   //forward
                test_net->Forward(&loss);   //forward
                test_loss += loss;

                //Calculate the landmarks difference:
                const float* target = label_blob->cpu_data();
                const float* actual = landmarks_blob->cpu_data();
                float points_acc = 0.0;
                float x_acc = 0.0, y_acc = 0.0;
                float contour_acc = 0.0, nose_acc = 0.0, mouth_acc = 0.0, eyes_acc = 0.0, eyebrows_acc = 0.0;

                for (int s = 0; s < num; ++s) {
                    for (int j = 0; j < label_size; j+=2)
                    {
                        int index = s*label_size + j;
                        float dif_x = pow(target[index] - actual[index], 2);
                        float dif_y = pow(target[index+1] - actual[index+1], 2);
                        x_acc += int(sqrt(dif_x) < acc_threshold);
                        y_acc += int(sqrt(dif_y) < acc_threshold);
                        points_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                    }
                    if (LABEL_TYPE == LABEL_LANDMARKS_68P) {
                        //Contour:
                        for (int j = CONTOUR_INDEX*2; j < (CONTOUR_INDEX+CONTOUR_COUNT)*2; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            contour_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Eyebrows:
                        for (int j = EYEBROWS_INDEX*2; j < (EYEBROWS_INDEX+EYEBROWS_COUNT)*2; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            eyebrows_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Nose:
                        for (int j = NOSE_INDEX*2; j < (NOSE_INDEX+NOSE_COUNT)*2; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            nose_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Eyes:
                        for (int j = EYES_INDEX*2; j < (EYES_INDEX+EYES_COUNT)*2; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            eyes_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Mouth:
                        for (int j = MOUTH_INDEX*2; j < (MOUTH_INDEX+MOUTH_COUNT)*2; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            mouth_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                    }
                    else if (LABEL_TYPE == LABEL_LANDMARKS_51P) {
                        const int offset = 17*2;

                        //Eyebrows:
                        for (int j = EYEBROWS_INDEX*2 - offset; j < (EYEBROWS_INDEX+EYEBROWS_COUNT)*2 - offset; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            eyebrows_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Nose:
                        for (int j = NOSE_INDEX*2 - offset; j < (NOSE_INDEX+NOSE_COUNT)*2 - offset; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            nose_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Eyes:
                        for (int j = EYES_INDEX*2 - offset; j < (EYES_INDEX+EYES_COUNT)*2 - offset; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            eyes_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                        //Mouth:
                        for (int j = MOUTH_INDEX*2 - offset; j < (MOUTH_INDEX+MOUTH_COUNT)*2 - offset; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            mouth_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                    }
                    else if (LABEL_TYPE == LABEL_LANDMARKS_17P) {
                        //Contour:
                        for (int j = CONTOUR_INDEX*2; j < (CONTOUR_INDEX+CONTOUR_COUNT)*2; j+=2) {
                            int index = s*label_size + j;
                            float dif_x = pow(target[index] - actual[index], 2);
                            float dif_y = pow(target[index+1] - actual[index+1], 2);
                            contour_acc += int(sqrt(dif_x + dif_y) < acc_threshold);
                        }
                    }
                }
                test_x_acc += x_acc/float(label_size*0.5*num);
                test_y_acc += y_acc/float(label_size*0.5*num);
                test_points_acc += points_acc/float(label_size*0.5*num);
                test_contour_acc += contour_acc/float(CONTOUR_COUNT*num);
                test_eyebrows_acc += eyebrows_acc/float(EYEBROWS_COUNT*num);
                test_nose_acc += nose_acc/float(NOSE_COUNT*num);
                test_eyes_acc += eyes_acc/float(EYES_COUNT*num);
                test_mouth_acc += mouth_acc/float(MOUTH_COUNT*num);

//                //DEBUG:
//                {
//                    const float* data = data_blob->cpu_data();
////                    const float* landmarks = landmarks_blob->cpu_data();
//                    const float* landmarks = label_blob->cpu_data();


//                    for (int j = 0; j < test_net_batch_size; ++j) {
//                        //Merge channels:
//                        cv::Mat out_img;
//                        std::vector<cv::Mat> channels(3);
//                        channels[0] = cv::Mat(160,160, CV_32FC1, const_cast<float*>(data) + 160*160*3*j);
//                        channels[1] = cv::Mat(160,160, CV_32FC1, const_cast<float*>(data) + 160*160*3*j + 160*160);
//                        channels[2] = cv::Mat(160,160, CV_32FC1, const_cast<float*>(data) + 160*160*3*j + 160*160*2);
//                        cv::merge(channels, out_img);
//                        out_img /= 0.00390625;

//                        //Draw landmarks:
//                        if (LABEL_TYPE == LABEL_LANDMARKS_68P || LABEL_TYPE == LABEL_LANDMARKS_51P || LABEL_TYPE == LABEL_LANDMARKS_17P) {
//                            int line_size = int(floor(out_img.cols/100.0*2.5/2.0 + 0.5));
//                            int color_step = floor(255.0/(label_size/2.0)+0.5);
//                            for (int k = 0; k < label_size; k+=2) {
//                                float x = landmarks[label_size*j + k];
//                                float y = landmarks[label_size*j + k+1];
//                                cv::Point2i point(int(floor(x*out_img.cols+0.5)), int(floor(y*out_img.rows+0.5)));
//                                cv::Point2i pt1(point.x-line_size, point.y);
//                                cv::Point2i pt2(point.x+line_size, point.y);
//                                cv::Point2i pt3(point.x, point.y-line_size);
//                                cv::Point2i pt4(point.x, point.y+line_size);
//                                cv::line(out_img, pt1, pt2, cv::Scalar(k/2*color_step, 255-k/2*color_step,255));
//                                cv::line(out_img, pt3, pt4, cv::Scalar(k/2*color_step, 255-k/2*color_step,255));
//                            }
//                        }
//                        else if (LABEL_TYPE == LABEL_RECT_2P) {
//                            cv::Point2i pt1(int(floor(landmarks[label_size*j+0]*out_img.cols+0.5)), int(floor(landmarks[label_size*j+1]*out_img.rows+0.5)));
//                            cv::Point2i pt3(int(floor(landmarks[label_size*j+2]*out_img.cols+0.5)), int(floor(landmarks[label_size*j+3]*out_img.rows+0.5)));
//                            cv::Point2i pt2(pt3.x, pt1.y);
//                            cv::Point2i pt4(pt1.x, pt3.y);
//                            cv::line(out_img, pt1, pt2, cv::Scalar(255,0,0));
//                            cv::line(out_img, pt2, pt3, cv::Scalar(0,255,0));
//                            cv::line(out_img, pt3, pt4, cv::Scalar(0,0,255));
//                            cv::line(out_img, pt4, pt1, cv::Scalar(255,0,255));
//                        }

//                        QString img_name = QString("/home/darkalert/Desktop/test/%1_%2.png").arg(i).arg(j);
//                        cv::imwrite(img_name.toStdString(), out_img);
//                    }
//                }
            }
            qDebug() << "Test phase, iteration" << solver->iter();
            float count = ceil(num_test_data/float(num));
            test_loss /= count;
            test_points_acc /= count;
            test_x_acc /= count;
            test_y_acc /= count;
            float test_xy_acc = (test_x_acc + test_y_acc)*0.5;
            test_contour_acc /= count;
            test_eyebrows_acc /= count;
            test_nose_acc /= count;
            test_eyes_acc /= count;
            test_mouth_acc /= count;
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

    train_log.close();
    train_worker_is_running = false;
}

void LandmarkTrainer::train(const std::string &path_to_solver)
{
    if (running == true) {
        qDebug() << "ClassifierTrainer: Training is already running.";
        return;
    }

    if (data_provider->total_train_samples() == 0) {
        qDebug() << "ClassifierTrainer: Training data are empty!";
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
    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/68p_resnet1/snaps/classifier_v2_iter_20000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/darkalert/Desktop/MirrorJob/Snaps/68p_resnet1/snaps/classifier_v1_iter_5000.caffemodel");

    //Log:
    qDebug() << "ClassifierTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "ClassifierTrainer: GPU mode has been set." : "ClassifierTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&LandmarkTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&LandmarkTrainer::train_by_worker, this);
    train_worker.detach();
}

void LandmarkTrainer::restore(const std::string &path_to_solver, const std::string &path_to_solverstate)
{
    if (running == true) {
        qDebug() << "ClassifierTrainer: Training is already running.";
        return;
    }

    if (data_provider->total_train_samples()) {
        qDebug() << "ClassifierTrainer: Training data are empty!";
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
    qDebug() << "ClassifierTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << "ClassifierTrainer: Solver's state has been restored from" << QString::fromStdString(path_to_solverstate);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "ClassifierTrainer: GPU mode has been set." : "ClassifierTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&LandmarkTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&LandmarkTrainer::train_by_worker, this);
    train_worker.detach();
}

void LandmarkTrainer::stop()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    if (data_provider != nullptr) {
        data_provider->stop();
        data_provider = nullptr;
    }

    //Snapshot:
    if (solver != nullptr) {
        solver->Snapshot();
    }

    solver = nullptr;
    test_net = nullptr;
    test_data.first = nullptr;
    test_data.second = nullptr;
    num_test_data = 0;
}

void LandmarkTrainer::pause()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    qDebug() << "ClassifierTrainer: Training is paused. Current iter:" << solver->iter();
}

void LandmarkTrainer::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&LandmarkTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&LandmarkTrainer::train_by_worker, this);
    train_worker.detach();

    qDebug() << "LandmarkTrainer: Training is resumed.";
}


