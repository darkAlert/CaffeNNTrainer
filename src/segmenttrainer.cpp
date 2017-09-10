#include "segmenttrainer.h"
#include "helper.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include "caffe/net.hpp"
#include <cuda.h>
#include <opencv2/cudawarping.hpp>

#include <QDebug>
#include <QTime>
#include <QFile>
#include <QFileInfo>

using namespace caffe;



SegmentTrainer::SegmentTrainer()
{
    solver = nullptr;
    running = false;
    prefetch_worker_is_running = false;
    train_worker_is_running = false;
    test_net = nullptr;
    test_data.first = nullptr;
    test_data.second = nullptr;
}

SegmentTrainer::~SegmentTrainer()
{
    stop();
}

void SegmentTrainer::prefetch_batches_by_worker(unsigned int batch_margin)
{
    prefetch_worker_is_running = true;

    //Batch params:
    const int batch_size = solver->net()->blob_by_name("data")->shape(0);
    const int sample_channels = solver->net()->blob_by_name("data")->shape(1);
    const int sample_height = solver->net()->blob_by_name("data")->shape(2);
    const int sample_width = solver->net()->blob_by_name("data")->shape(3);
    const int sample_size = sample_channels*sample_height*sample_width;
    const int label_channels = solver->net()->blob_by_name("label")->shape(1);
    const int label_height = solver->net()->blob_by_name("label")->shape(2);
    const int label_width = solver->net()->blob_by_name("label")->shape(3);
    const int label_size = label_channels*label_height*label_width;

    struct Distortions {
        const bool crop = true;
        const cv::Size2i max_crop_offset = cv::Size2i(160,160);
//        const cv::Size2i max_crop_offset = cv::Size2i(120,160);

        const bool mirror = true;

        const bool rotate = true;
        const float max_rotate_angle = 20.0;

        const bool scale = false;
        const float max_scale_ratio = 0.05;

        const bool deform = true;
        const float max_deform_ratio = 1.2;

        const bool grayscale = true;
        const float grayscale_prob = 0.2;

        const bool jpeg = true;
        const float jpeg_min_quality = 12;
        const float jpeg_max_quality = 20;

        const bool brightness = true;
        const float bright_min_alpha = 0.1;
        const float bright_max_alpha = 2.0;
        const float bright_min_beta = -40/255.0;
        const float bright_max_beta = 40/255.0;
    } DISTORTS;

    //Distributions:
    srand(std::time(nullptr));
    rng.seed(std::time(nullptr));
    boost::random::uniform_real_distribution<float> die_one(0.0, 1.0);
    boost::random::uniform_real_distribution<float> die_rotate(-DISTORTS.max_rotate_angle, DISTORTS.max_rotate_angle);
    boost::random::uniform_real_distribution<float> die_scale(0.0, DISTORTS.max_scale_ratio);
    boost::random::uniform_real_distribution<float> die_deform(1.0, DISTORTS.max_deform_ratio);
    boost::random::uniform_int_distribution<int> die_jpeg(DISTORTS.jpeg_min_quality, DISTORTS.jpeg_max_quality);
    boost::random::uniform_real_distribution<float> die_bright_alpha(DISTORTS.bright_min_alpha, DISTORTS.bright_max_alpha);
    boost::random::uniform_real_distribution<float> die_bright_beta(DISTORTS.bright_min_beta, DISTORTS.bright_max_beta);

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
                /// Get index ///
                unsigned int index = train_samples_indices[current_train_index];
                ++current_train_index;
                if (current_train_index >= train_samples_indices.size()) {
                    //Shuffle:
                    uint32_t times = train_samples_indices.size();
                    boost::random::uniform_int_distribution<uint64_t> die_index = boost::random::uniform_int_distribution<uint64_t> (0, train_samples_indices.size() - 1);
                    for (uint32_t k = 0; k < times; ++k) {
                        auto i1 = die_index(rng);
                        auto i2 = die_index(rng);
                        if (i1 == i2) continue;
                        std::swap(train_samples_indices[i1], train_samples_indices[i2]);
                    }
                    current_train_index = 0;
                }

                /// Open sample and label ///
                const auto& raw_sample = raw_train_samples[index].first;
                const auto& raw_label = raw_train_samples[index].second;
                cv::Mat sample_float, label_float;
                int type_i = raw_sample.channels() == 1 ? CV_8UC1 : CV_8UC3;
                int type_f = raw_sample.channels() == 1 ? CV_32FC1 : CV_32FC3;
                cv::Mat(raw_sample.rows, raw_sample.cols, type_i, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, type_f, 0.00390625);
                type_i = raw_label.channels() == 1 ? CV_8UC1 : CV_8UC3;
                type_f = raw_label.channels() == 1 ? CV_32FC1 : CV_32FC3;
//                cv::Mat(raw_label.rows, raw_label.cols, type_i, const_cast<unsigned char*>(raw_label.data)).convertTo(label_float, type_f, 0.00390625);
                cv::Mat(raw_label.rows, raw_label.cols, type_i, const_cast<unsigned char*>(raw_label.data)).convertTo(label_float, type_f, 1.0);

                /// Augmentation ///
                //Deformation:
                if (DISTORTS.deform && die_one(rng) < 0.5)
                {
                    float ratio = die_deform(rng);
                    if (die_one(rng) < 0.5) {
                        cv::resize(sample_float, sample_float, cv::Size(0,0), ratio, 1.0, CV_INTER_LINEAR);
                        cv::resize(label_float, label_float, cv::Size(0,0), ratio, 1.0, CV_INTER_NN);
                    }
                    else {
                        cv::resize(sample_float, sample_float, cv::Size(0,0), 1.0, ratio, CV_INTER_LINEAR);
                        cv::resize(label_float, label_float, cv::Size(0,0), 1.0, ratio, CV_INTER_NN);
                    }
                }
                //Rotate:
                if (DISTORTS.rotate) {
                    float angle = die_rotate(rng);
                    auto rot_mat = cv::getRotationMatrix2D(cv::Point(floor(sample_float.cols*0.5+0.5), floor(sample_float.rows*0.5+0.5)), angle, 1.0);
                    if (gpu_mode == true) {
                        {
                            //Sample:
                            cv::cuda::GpuMat frame_gpu(sample_float);
                            cv::cuda::GpuMat frame_rotated_gpu;
                            cv::cuda::warpAffine(frame_gpu, frame_rotated_gpu, rot_mat, cv::Size(sample_float.cols, sample_float.rows));
                            frame_rotated_gpu.download(sample_float);
                        }
                        {
                            //Label:
                            cv::cuda::GpuMat frame_gpu(label_float);
                            cv::cuda::GpuMat frame_rotated_gpu;
                            cv::cuda::warpAffine(frame_gpu, frame_rotated_gpu, rot_mat, cv::Size(label_float.cols, label_float.rows), cv::INTER_NEAREST);
                            frame_rotated_gpu.download(label_float);
                        }

                    }
                    else {
                        cv::warpAffine(sample_float, sample_float, rot_mat, cv::Size(sample_float.cols, sample_float.rows));
                        cv::warpAffine(label_float, label_float, rot_mat, cv::Size(label_float.cols, label_float.rows), cv::INTER_NEAREST);
                    }
                }
                //Crop:
                if (DISTORTS.crop && (sample_float.cols != sample_width || sample_float.rows != sample_height)) {
                    int dif_w = std::min(sample_float.cols,DISTORTS.max_crop_offset.width) - sample_width;
                    int dif_h = std::min(sample_float.rows,DISTORTS.max_crop_offset.height) - sample_height;
                    cv::Rect crop_rect(0,0,sample_width,sample_height);
                    crop_rect.x = rand()%(dif_w+1) + std::max(int(floor((sample_float.cols-DISTORTS.max_crop_offset.width)*0.5+0.5)),0);
                    crop_rect.y = rand()%(dif_h+1) + std::max(int(floor((sample_float.rows-DISTORTS.max_crop_offset.height)*0.5+0.5)),0);
                    sample_float = sample_float(crop_rect).clone();
                    label_float = label_float(crop_rect).clone();
                }
                //Scale:
                if (DISTORTS.scale) {
                    float ratio = 1.0 + die_scale(rng);
                    cv::Mat scaled_sample, scaled_label;
                    cv::resize(sample_float, scaled_sample, cv::Size(0,0), ratio, ratio, CV_INTER_CUBIC);
                    cv::resize(label_float, scaled_label, cv::Size(0,0), ratio, ratio, cv::INTER_NEAREST);
                    cv::Rect crop_rect(floor((scaled_sample.cols-sample_width)*0.5 + 0.5), floor((scaled_sample.rows-sample_height)*0.5 + 0.5), sample_width, sample_height);
                    sample_float = scaled_sample(crop_rect).clone();
                    label_float = scaled_label(crop_rect).clone();
                }
                //Mirror:
                if (DISTORTS.mirror && rand() < RAND_MAX/2) {
                    cv::flip(sample_float,sample_float,1);
                    cv::flip(label_float,label_float,1);
                }
                if (DISTORTS.grayscale && die_one(rng) < DISTORTS.grayscale_prob) {
                    if (sample_float.channels() == 3) {
                        cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2GRAY);
                    }
                    cv::cvtColor(sample_float, sample_float, cv::COLOR_GRAY2BGR);
                }
                //Resize to fit:
                if (sample_float.rows != sample_height || sample_float.cols != sample_width) {
                    auto inter = sample_float.cols > sample_width ? CV_INTER_AREA : CV_INTER_LINEAR;
                    cv::resize(sample_float,sample_float,cv::Size(sample_width, sample_height), 0, 0, inter);
                }
                if (label_float.rows != label_height || label_float.cols != label_width) {
                    auto inter = cv::INTER_NEAREST;//label_float.cols > label_width ? CV_INTER_AREA : CV_INTER_LINEAR;
                    cv::resize(label_float,label_float,cv::Size(label_width, label_height), 0, 0, inter);
                }
                //Brightness:
                if (DISTORTS.brightness && rand() < RAND_MAX/2)
                {
                    sample_float = sample_float*die_bright_alpha(rng) + die_bright_beta(rng);
                }
                //JPEG:
                if (DISTORTS.jpeg && rand() < RAND_MAX/2)
                {
                    //Compression params:
                    std::vector<int> param(2);
                    param[0] = cv::IMWRITE_JPEG_QUALITY;
                    param[1] = die_jpeg(rng);

                    //Float to int:
                    cv::Mat sample_int;
                    const int type_i = CV_8UC(sample_float.channels());
                    sample_float.convertTo(sample_int, type_i,  255.0);

                    //Compress:
                    std::vector<unsigned char> mat_encoded;
                    cv::imencode(".jpeg", sample_int, mat_encoded, param);

                    //Uncompress:
                    sample_int = cv::imdecode(mat_encoded, -1);

                    //Int to float:
                    const int type_f = CV_32FC(sample_int.channels());
                    sample_int.convertTo(sample_float, type_f,  0.00390625);
                }

                /// Set data to the net ///
                assert(sample_channels <= sample_float.channels());
                if (sample_channels == 1) {
                    memcpy(&(data_batch.get()[i*sample_size]), sample_float.data, sizeof(float)*(sample_size));
                }
                else {
                    std::vector<cv::Mat> channels;
                    channels.resize(sample_float.channels());
                    cv::split(sample_float,channels);
                    int channel_size = sample_size/sample_channels;
                    for (int ci = 0; ci < sample_channels; ++ci) {
                        memcpy(&(data_batch.get()[i*sample_size+channel_size*ci]), channels[ci].data, sizeof(float)*(channel_size));
                    }
                }
                if (label_channels == 1) {
                    memcpy(&(labels_batch.get()[i*label_size]), label_float.data, sizeof(float)*(label_size));
                }
                else {
                    std::vector<cv::Mat> channels;
                    channels.resize(label_float.channels());
                    cv::split(label_float,channels);
                    int channel_size = label_size/label_channels;
                    for (int ci = 0; ci < label_channels; ++ci) {
                        memcpy(&(labels_batch.get()[i*label_size+channel_size*ci]), channels[ci].data, sizeof(float)*(channel_size));
                    }
                }
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

void SegmentTrainer::prepare_test_data()
{
    if (test_net == nullptr) return;
    assert(num_test_samples <= raw_test_samples.size());

    //Data params:
    const int batch_size = test_net->blob_by_name("data")->shape(0);
    const int sample_channels = test_net->blob_by_name("data")->shape(1);
    const int sample_height = test_net->blob_by_name("data")->shape(2);
    const int sample_width = test_net->blob_by_name("data")->shape(3);
    const int sample_size = sample_channels*sample_height*sample_width;
    const int label_channels = test_net->blob_by_name("label")->shape(1);
    const int label_height = test_net->blob_by_name("label")->shape(2);
    const int label_width = test_net->blob_by_name("label")->shape(3);
    const int label_size = label_channels*label_height*label_width;
    assert(int(num_test_samples) <= batch_size);

    //Allocate memory:
    test_data.first.reset(new float[num_test_samples*sample_size]);
    test_data.second.reset(new float[num_test_samples*label_size]);

    //Prepare data:
    for (unsigned int i = 0; i < num_test_samples; ++i)
    {
        //Open:
        const auto& raw_sample = raw_test_samples[i].first;
        const auto& raw_label = raw_test_samples[i].second;
        cv::Mat sample_float, label_float;
        int type_i = raw_sample.channels() == 1 ? CV_8UC1 : CV_8UC3;
        int type_f = raw_sample.channels() == 1 ? CV_32FC1 : CV_32FC3;
        cv::Mat(raw_sample.rows, raw_sample.cols, type_i, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, type_f, 0.00390625);
        type_i = raw_label.channels() == 1 ? CV_8UC1 : CV_8UC3;
        type_f = raw_label.channels() == 1 ? CV_32FC1 : CV_32FC3;
//        cv::Mat(raw_label.rows, raw_label.cols, type_i, const_cast<unsigned char*>(raw_label.data)).convertTo(label_float, type_f, 0.00390625);
        cv::Mat(raw_label.rows, raw_label.cols, type_i, const_cast<unsigned char*>(raw_label.data)).convertTo(label_float, type_f, 1.0);

        //Crop:
        if (sample_float.cols != sample_width || sample_float.rows != sample_height) {
            int dif_w = sample_float.cols - sample_width;
            int dif_h = sample_float.rows - sample_height;
            cv::Rect crop_rect(0,0,sample_width,sample_height);
            crop_rect.x = floor(dif_w*0.5 + 0.5);
            crop_rect.y = floor(dif_h*0.5 + 0.5);
            sample_float = sample_float(crop_rect).clone();
            label_float = label_float(crop_rect).clone();
        }

        //Resize to fit:
        if (sample_float.rows != sample_height || sample_float.cols != sample_width) {
            auto inter = sample_float.cols > sample_width ? CV_INTER_AREA : CV_INTER_LINEAR;
            cv::resize(sample_float,sample_float,cv::Size(sample_width, sample_height), 0, 0, inter);
        }
        if (label_float.rows != label_height || label_float.cols != label_width) {
            auto inter = cv::INTER_NEAREST;//label_float.cols > label_width ? CV_INTER_AREA : CV_INTER_LINEAR;
            cv::resize(label_float,label_float,cv::Size(label_width, label_height), 0, 0, inter);
        }

        //Set data to the net:
        assert(sample_channels <= sample_float.channels());
        if (sample_channels == 1) {
            memcpy(&(test_data.first.get()[i*sample_size]), sample_float.data, sizeof(float)*(sample_size));
        }
        else {
            std::vector<cv::Mat> channels;
            channels.resize(sample_float.channels());
            cv::split(sample_float,channels);
            int channel_size = sample_size/sample_channels;
            for (int ci = 0; ci < sample_channels; ++ci) {
                memcpy(&(test_data.first.get()[i*sample_size+channel_size*ci]), channels[ci].data, sizeof(float)*(channel_size));
            }
        }
        if (label_channels == 1) {
            memcpy(&(test_data.second.get()[i*label_size]), label_float.data, sizeof(float)*(label_size));
        }
        else {
            std::vector<cv::Mat> channels;
            channels.resize(label_float.channels());
            cv::split(label_float,channels);
            int channel_size = label_size/label_channels;
            for (int ci = 0; ci < label_channels; ++ci) {
                memcpy(&(test_data.second.get()[i*label_size+channel_size*ci]), channels[ci].data, sizeof(float)*(channel_size));
            }
        }
    }
}

void SegmentTrainer::set_train_batch()
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

void SegmentTrainer::set_test_batch(int sample_index)
{
    assert(test_net != nullptr);
    boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    assert((unsigned int)num == num_test_samples);
    const int sample_size = data_blob->shape(3) * data_blob->shape(2) * data_blob->shape(1);
    const int label_size = label_blob->shape(3) * label_blob->shape(2) * label_blob->shape(1);
    float* data = data_blob->mutable_cpu_data();
    float* labels = label_blob->mutable_cpu_data();

    //Set the data and labels:
    auto rest = num_test_samples-sample_index >= (unsigned int)num ? (unsigned int)num : num_test_samples-sample_index;
    memcpy(data, &(test_data.first.get()[sample_index*sample_size]), sizeof(float)*sample_size*rest);
    memcpy(labels, &(test_data.second.get()[sample_index*label_size]), sizeof(float)*label_size*rest);
}

bool SegmentTrainer::read_raw_samples(const std::string &path_to_csv)
{
    //Open csv containing dirs:
    std::ifstream csv_file(path_to_csv.c_str(), std::ifstream::in);
    if (!csv_file) {
        qDebug() << "SegmentTrainer: No valid input file was given, please check the given filename.";
        return false;
    }
    QString path_to_dir = QFileInfo(QFile(QString::fromStdString(path_to_csv))).absolutePath() + "/";

    //Allocate memory:
    raw_train_samples.clear();
    raw_test_samples.clear();
    raw_test_samples.reserve(num_test_samples);
    raw_train_samples.reserve(std::count(std::istreambuf_iterator<char>(csv_file), std::istreambuf_iterator<char>(), '\n') - num_test_samples);
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

        //Open images:
        cv::Mat sample = cv::imread(path_to_sample, -1);
        cv::Mat label = cv::imread(path_to_label, -1);
        if (sample.channels() > 3) {
            cv::cvtColor(sample, sample, CV_BGRA2BGR);
        }

        //Push sample and label:
        if (count < num_test_samples) {
            raw_test_samples.push_back(std::make_pair(sample.clone(), label.clone()));
        }
        else {
            raw_train_samples.push_back(std::make_pair(sample.clone(), label.clone()));
        }
        ++count;
    }
    assert(raw_test_samples.size() == num_test_samples);

    //Make indices array:
    train_samples_indices.resize(raw_train_samples.size());
    for (unsigned int i = 0; i < train_samples_indices.size(); ++i) {
        train_samples_indices[i] = i;
    }
    current_train_index = 0;

    qDebug() << "SegmentTrainer: Sample have been loaded. Train sampels:" << raw_train_samples.size() << ", test samples:" << raw_test_samples.size();
    return true;
}

void SegmentTrainer::train_by_worker()
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
            Caffe::set_mode(Caffe::GPU);
            set_test_batch(0);
            float loss = 0.0;
            float acc = 0.0;
            const std::vector<Blob<float>*>& result = test_net->Forward(&loss);   //forward
            acc += result[0]->cpu_data()[0];

            qDebug() << "Test phase, iteration" << solver->iter();
            qDebug() << "   Test net: loss =" << loss << ", acc =" << acc;
/*
            //DEBUG:
            {
                boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
                boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
                boost::shared_ptr<Blob<float> > output_blob = test_net->blob_by_name("softmax");
                const float* data = data_blob->cpu_data();
                const float* labels = label_blob->cpu_data();
                const float* output = output_blob->cpu_data();
                const int batch_size = test_net->blob_by_name("data")->shape(0);
                const int sample_channels = test_net->blob_by_name("data")->shape(1);
                const int sample_height = test_net->blob_by_name("data")->shape(2);
                const int sample_width = test_net->blob_by_name("data")->shape(3);
                const int sample_size = sample_channels*sample_height*sample_width;
                const int label_channels = test_net->blob_by_name("label")->shape(1);
                const int label_height = test_net->blob_by_name("label")->shape(2);
                const int label_width = test_net->blob_by_name("label")->shape(3);
                const int label_size = label_channels*label_height*label_width;
                const int output_channels = test_net->blob_by_name("output")->shape(1);
                const int output_height = test_net->blob_by_name("output")->shape(2);
                const int output_width = test_net->blob_by_name("output")->shape(3);
                const int output_size = output_channels*output_height*output_width;

                for (int j = 0; j < batch_size; ++j) {
                    cv::Mat img;
                    std::vector<cv::Mat> channels(3);
                    channels[0] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j);
                    channels[1] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width);
                    channels[2] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width*2);
                    cv::merge(channels, img);
                    img /= 0.00390625;
                    QString path = QString("/home/darkalert/Desktop/test/%1_s.png").arg(j);
                    cv::imwrite(path.toStdString(), img);

                    cv::Mat label;
                    channels.resize(3);
                    channels[0] = cv::Mat(label_height,label_width, CV_32FC1, const_cast<float*>(labels) + label_size*j);
//                    channels[1] = cv::Mat(label_height,label_width, CV_32FC1, const_cast<float*>(labels) + label_size*j + label_height*label_width);
                    channels[1] = cv::Mat(label_height,label_width, CV_32FC1);
                    channels[1] = cv::Scalar(0,0,0);
                    channels[2] = cv::Mat(label_height,label_width, CV_32FC1);
                    channels[2] = cv::Scalar(0,0,0);
                    cv::merge(channels, label);
                    label /= 0.00390625;
                    QString path2 = QString("/home/darkalert/Desktop/test/%1_l.png").arg(j);
                    cv::imwrite(path2.toStdString(), label);

                    for (unsigned int i = 1; i < 2; ++i) {
                        float* ptr =  const_cast<float*>(output) + output_size*j + output_height*output_width*i;
                        cv::Mat mask;
                        cv::Mat(output_height, output_width, CV_32F, ptr).convertTo(mask, CV_8UC1, 1);
                        cv::Mat segment;
                        cv::bitwise_and(img,img,segment,mask);
                        QString path = QString("/home/darkalert/Desktop/test/%1_o%2.png").arg(j).arg(i);
                        cv::imwrite(path.toStdString(), segment);
                    }
                }
            }
            */
        }

        /// Treain phase ///
        Caffe::set_mode(Caffe::GPU);
        set_train_batch();
        solver->Step(1);
/*
        //DEBUG:
        {
            auto net = solver->net();
            boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
            boost::shared_ptr<Blob<float> > label_blob = net->blob_by_name("label");
            const float* data = data_blob->cpu_data();
            const float* labels = label_blob->cpu_data();
            const int batch_size = net->blob_by_name("data")->shape(0);
            const int sample_channels = net->blob_by_name("data")->shape(1);
            const int sample_height = net->blob_by_name("data")->shape(2);
            const int sample_width = net->blob_by_name("data")->shape(3);
            const int sample_size = sample_channels*sample_height*sample_width;
            const int label_channels = net->blob_by_name("label")->shape(1);
            const int label_height = net->blob_by_name("label")->shape(2);
            const int label_width = net->blob_by_name("label")->shape(3);
            const int label_size = label_channels*label_height*label_width;

            for (int j = 0; j < batch_size; ++j) {
                cv::Mat out_img;
                std::vector<cv::Mat> channels(3);
                channels[0] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j);
                channels[1] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width);
                channels[2] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width*2);
                cv::merge(channels, out_img);
                out_img /= 0.00390625;
                QString path = QString("/home/darkalert/Desktop/test/%1_s.png").arg(j);
                cv::imwrite(path.toStdString(), out_img);

                channels.resize(3);
                channels[0] = cv::Mat(label_height,label_width, CV_32FC1, const_cast<float*>(labels) + label_size*j);
//                channels[1] = cv::Mat(label_height,label_width, CV_32FC1, const_cast<float*>(labels) + label_size*j + label_height*label_width);
                channels[1] = cv::Mat(label_height,label_width, CV_32FC1);
                channels[1] = cv::Scalar(0,0,0);
                channels[2] = cv::Mat(label_height,label_width, CV_32FC1);
                channels[2] = cv::Scalar(0,0,0);
                cv::merge(channels, out_img);
                out_img /= 0.00390625;
                QString path2 = QString("/home/darkalert/Desktop/test/%1_l.png").arg(j);
                cv::imwrite(path2.toStdString(), out_img);
            }
        }*/
    }

    train_worker_is_running = false;
}

void SegmentTrainer::train(const std::string &path_to_solver)
{
    if (running == true) {
        qDebug() << "SegmentTrainer: Training is already running.";
        return;
    }

    if (raw_train_samples.empty()) {
        qDebug() << "SegmentTrainer: Training data are empty!";
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
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, num_test_samples);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(1)->set_dim(0, num_test_samples);
    test_net_param.mutable_state()->set_phase(Phase::TEST);
    test_net = std::make_shared<Net<float> >(test_net_param);
    solver->net().get()->ShareTrainedLayersWith(test_net.get());

    //Load weights:
//    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/segment_resnet1/best/resnet9_l9_v4_iter_5000.caffemodel");
    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/segment_resnet1/best/resnet11_l9_v5_iter_15000.caffemodel");

    //Log:
    qDebug() << "SegmentTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "SegmentTrainer: GPU mode has been set." : "SegmentTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&SegmentTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&SegmentTrainer::train_by_worker, this);
    train_worker.detach();
}

void SegmentTrainer::restore(const std::string &path_to_solver, const std::string &path_to_solverstate)
{
    if (running == true) {
        qDebug() << "SegmentTrainer: Training is already running.";
        return;
    }

    if (raw_train_samples.empty()) {
        qDebug() << "SegmentTrainer: Training data are empty!";
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
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, num_test_samples);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(1)->set_dim(0, num_test_samples);
    test_net_param.mutable_state()->set_phase(Phase::TEST);
    test_net = std::make_shared<Net<float> >(test_net_param);
    solver->net().get()->ShareTrainedLayersWith(test_net.get());

    //Restore state:
    solver->Restore(path_to_solverstate.c_str());

    //Log:
    qDebug() << "SegmentTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << "SegmentTrainer: Solver's state has been restored from" << QString::fromStdString(path_to_solverstate);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "SegmentTrainer: GPU mode has been set." : "SegmentTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&SegmentTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&SegmentTrainer::train_by_worker, this);
    train_worker.detach();
}

void SegmentTrainer::stop()
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

void SegmentTrainer::pause()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        helper::little_sleep(std::chrono::microseconds(100));
    }

    qDebug() << "ClassifierTrainer: Training is paused. Current iter:" << solver->iter();
}

void SegmentTrainer::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&SegmentTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&SegmentTrainer::train_by_worker, this);
    train_worker.detach();

    qDebug() << "SegmentTrainer: Training is resumed.";
}
