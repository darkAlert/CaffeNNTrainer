#include "attributepredictortrainer.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include <QFileInfo>
#include <QDebug>
#include <QTime>
#include <QDir>

#include "caffe/net.hpp"
#include "helper.h"
#include <cuda.h>
#include <opencv2/cudawarping.hpp>

using namespace caffe;

#define AT_BOOL 0
#define AT_RECOLORAZING_FLOAT 1
#define AT_RECOLORAZING_SOFTMAX 2
#define ATTRIBUTES_TEST_TYPE 5


AttributePredictorTrainer::AttributePredictorTrainer(bool gpu_mode)
    : gpu_mode(gpu_mode)
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

AttributePredictorTrainer::~AttributePredictorTrainer()
{
    stop();
}

void AttributePredictorTrainer::prefetch_batches_by_worker(unsigned int batch_margin)
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
//        const cv::Size2i max_crop_offset = cv::Size2i(150,200);      //landmarks68p
//         const cv::Size2i max_crop_offset = cv::Size2i(252,252);
        const cv::Size2i max_crop_offset = cv::Size2i(144,144);
//        const cv::Size2i max_crop_offset = cv::Size2i(150,150);      //attributes

        const bool mirror = true;

        const bool scale = true;
        const float max_scale_ratio = 0.25;

        const bool rotate = true;
        const float max_rotate_angle = 10;//25.0;

        const bool blackbox = true;
        const float max_blackbox_size = 0.5;

        const bool deform = true;
        const float max_deform_ratio = 1.1;

        const bool grayscale = false;
        const float grayscale_prob = 0.5;

        const bool intensity_shift = true;
        const float max_intensity_shift = 30/255.0;

        const bool negative = false;
        const float negative_prob = 0.25;
    } DISTORTS;

    //Distributions:
    boost::random::uniform_real_distribution<float> die_one(0.0, 1.0);
    boost::random::uniform_real_distribution<float> die_scale(0.0, DISTORTS.max_scale_ratio);
    boost::random::uniform_real_distribution<float> die_rotate(-DISTORTS.max_rotate_angle, DISTORTS.max_rotate_angle);
    boost::random::uniform_real_distribution<float> die_deform(1.0, DISTORTS.max_deform_ratio);
    boost::random::uniform_real_distribution<float> die_blackbox(0.0, DISTORTS.max_blackbox_size);
    boost::random::uniform_real_distribution<float> die_intensity(-DISTORTS.max_intensity_shift, DISTORTS.max_intensity_shift);
    srand(std::time(nullptr));
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

            /// Pick samples ///
            for (int i = 0; i < batch_size; ++i)
            {
                std::vector<float> label;
                auto raw_sample = data_provider->get_train_sample(label);
                cv::Mat sample_float;
                int type_i = raw_sample.channels() == 1 ? CV_8UC1 : CV_8UC3;
                int type_f = raw_sample.channels() == 1 ? CV_32FC1 : CV_32FC3;
                cv::Mat(raw_sample.rows, raw_sample.cols, type_i, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, type_f, 0.00390625);

                //Deformation:
                if (DISTORTS.deform && rand() < RAND_MAX*0.5)
                {
                    float ratio = die_deform(rng);
                    if (rand() < RAND_MAX*0.5) {
                        cv::resize(sample_float, sample_float, cv::Size(0,0), ratio, 1.0, CV_INTER_LINEAR);
                    }
                    else {
                        cv::resize(sample_float, sample_float, cv::Size(0,0), 1.0, ratio, CV_INTER_LINEAR);
                    }
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
                }
                //Crop:
                if (DISTORTS.crop && (sample_float.cols != width || sample_float.rows != height)) {
                    int dif_w = std::min(sample_float.cols,DISTORTS.max_crop_offset.width) - width;
                    int dif_h = std::min(sample_float.rows,DISTORTS.max_crop_offset.height) - height;
                    cv::Rect crop_rect(0,0,width,height);
                    crop_rect.x = rand()%(dif_w+1) + std::max(int(floor((sample_float.cols-DISTORTS.max_crop_offset.width)*0.5+0.5)),0);
                    crop_rect.y = rand()%(dif_h+1) + std::max(int(floor((sample_float.rows-DISTORTS.max_crop_offset.height)*0.5+0.5)),0);
                    sample_float = sample_float(crop_rect).clone();
                }
                //Scale:
                if (DISTORTS.scale) {
                    float ratio = 1.0 + die_scale(rng);
                    cv::Mat scaled_sample;
                    cv::resize(sample_float, scaled_sample, cv::Size(0,0), ratio, ratio, CV_INTER_CUBIC);
                    cv::Rect crop_rect(floor((scaled_sample.cols-width)*0.5 + 0.5), floor((scaled_sample.rows-height)*0.5 + 0.5), width, height);
                    sample_float = scaled_sample(crop_rect).clone();
                }
                //Mirror:
                if (DISTORTS.mirror && die_one(rng) < 0.5) {
                    cv::flip(sample_float,sample_float,1);
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
                if (DISTORTS.grayscale && die_one(rng) < DISTORTS.grayscale_prob) {
                    if (sample_float.channels() == 3) {
                        cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2GRAY);
                    }
                    cv::cvtColor(sample_float, sample_float, cv::COLOR_GRAY2BGR);
                }
                //Resize to fit:
                if (sample_float.rows != height || sample_float.cols != width) {
                    auto inter = sample_float.cols > width ? CV_INTER_AREA : CV_INTER_LINEAR;
                    cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, inter);
                }
                if (DISTORTS.negative && sample_float.channels() == 1 && die_one(rng) < DISTORTS.negative_prob) {
                    sample_float = 1.0 - sample_float;
                    cv::threshold(sample_float, sample_float, 0.9999, 0.0, CV_THRESH_TOZERO_INV);
                }
                if (DISTORTS.intensity_shift && sample_float.channels() == 1) {
                    float shift = die_intensity(rng);
                    sample_float = sample_float + (sample_float/sample_float)*shift;
                    cv::threshold(sample_float, sample_float, 0.0, 0.0, CV_THRESH_TOZERO);
                    cv::threshold(sample_float, sample_float, 1.0, 1.0, CV_THRESH_TRUNC);
                }

                //Set data to the main net:
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

void AttributePredictorTrainer::prepare_test_data()
{
    if (test_net == nullptr) return;

    //Data params:
    const int channels =test_net->blob_by_name("data")->shape(1);
    const int height = test_net->blob_by_name("data")->shape(2);
    const int width = test_net->blob_by_name("data")->shape(3);
    const int sample_size = channels*height*width;
    const int label_size = test_net->blob_by_name("label")->shape(1) * test_net->blob_by_name("label")->shape(2) * test_net->blob_by_name("label")->shape(3);
    num_test_data = data_provider->total_test_samples();

    //Allocate memory:
    test_data.first.reset(new float[num_test_data*sample_size]);
    test_data.second.reset(new float[num_test_data*label_size]);

    //Prepare data:
    for (unsigned int i = 0; i < num_test_data; ++i)
    {
        std::vector<float> label;
        auto raw_sample = data_provider->get_test_sample(i, label);
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
            sample_float = sample_float(crop_rect).clone();
        }
        //Resize to fit:
        if (sample_float.rows != height || sample_float.cols != width) {
            auto inter = sample_float.cols > width ? CV_INTER_AREA : CV_INTER_LINEAR;
            cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, inter);
        }
        //Set data to the main net:
        if (sample_float.channels() == 1) {
            //Grayscale:
            memcpy(&(test_data.first.get())[i*sample_size], sample_float.data, sizeof(float)*(sample_size));
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

    data_provider->clear_test_batch();
}

void AttributePredictorTrainer::set_data_batch()
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

void AttributePredictorTrainer::set_test_data_batch(int sample_index)
{
    assert(test_net != nullptr);
    boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    const int sample_size = data_blob->shape(3) * data_blob->shape(2) * data_blob->shape(1);
    const int label_size = label_blob->shape(3) * label_blob->shape(2) * label_blob->shape(1);
    float* data = data_blob->mutable_cpu_data();
    float* labels = label_blob->mutable_cpu_data();
    assert(num == test_net_batch_size);

    //Set the data and labels:
    auto rest = (unsigned int)(num_test_samples)-sample_index >= (unsigned int)num ? (unsigned int)num : data_provider->total_test_samples()-sample_index;
    memcpy(data, &(test_data.first.get()[sample_index*sample_size]), sizeof(float)*sample_size*rest);
    memcpy(labels, &(test_data.second.get()[sample_index*label_size]), sizeof(float)*label_size*rest);
}

void AttributePredictorTrainer::openTrainData(const std::string &path_to_train)
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

void AttributePredictorTrainer::train_by_worker()
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
#if ATTRIBUTES_TEST_TYPE == AT_BOOL
        if (solver->iter() % test_interval == 0)
        {
            Caffe::set_mode(Caffe::GPU);
            const int num = test_net->blob_by_name("data")->shape(0);
            float test_loss = 0.0;
            boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
            boost::shared_ptr<Blob<float> > attr_blob = test_net->blob_by_name("output");
            const int label_size = label_blob->shape(1)*label_blob->shape(2)*label_blob->shape(3);
            const int attr_size = attr_blob->shape(1)*attr_blob->shape(2)*attr_blob->shape(3);
            float test_attr_accuracy = 0.0;

            for (unsigned int i = 0; i < num_test_data; i += num) {
                set_test_data_batch(i);
                float loss;
                test_net->Forward(&loss);   //forward
                test_loss += loss;

                //Calculate attributes difference:
                const float* labels = label_blob->cpu_data();
                const float* attributes = attr_blob->cpu_data();
                float attr_acc = 0.0;
                for (int s = 0; s < num; ++s) {
                    for (int j = 0; j < label_size; ++j) {
                        int index_label = s*label_size + j;
                        int index_attr = s*attr_size + j + label_size;
                        float attr = attributes[index_attr] >= 0.5 ? 1.0 : 0.0;
                        attr_acc += (attr == labels[index_label]  ? 1.0 : 0.0);
//                        if (s == 0) {
//                            qDebug() << j << ":" << out << "(" << labels[index] << "), true=" << (fabs(labels[index] - out) <= 0.5 ? 1.0 : 0.0);
//                        }
                    }
                }
                test_attr_accuracy += attr_acc/float(label_size*num);
            }
            qDebug() << "Test phase, iteration" << solver->iter();
            test_loss /= ceil(num_test_data/num);
            test_attr_accuracy /= ceil(num_test_data/num);
            qDebug() << "   Test net: Acc =" << test_attr_accuracy <<  ", loss =" << test_loss;
        }
#elif ATTRIBUTES_TEST_TYPE == AT_RECOLORAZING_FLOAT
        /// Test phase ///
        if (solver->iter() % test_interval == 0)
        {
            const float acc_threshold_value = 128.0;   // in color value
            const float acc_threshold_mul = 5;
            const float acc_threshold = 1.0/acc_threshold_value * acc_threshold_mul;
            Caffe::set_mode(Caffe::GPU);
//            boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
            boost::shared_ptr<Blob<float> > output_blob = test_net->blob_by_name("output");
            boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
            const int num = test_net->blob_by_name("data")->shape(0);
            const int label_size = output_blob->shape(1);
            float test_loss = 0.0;
            float test_acc = 0.0;
            float test_zero_acc = 0.0;
            float test_nonzero_acc = 0.0;

            for (unsigned int i = 0; i < num_test_data; i += num) {
                set_test_data_batch(i);
                float loss = 0.0;
                test_net->Forward(&loss);   //forward
                test_loss += loss;

                //Calculate the landmarks difference:
                const float* target = label_blob->cpu_data();
                const float* actual = output_blob->cpu_data();
                float acc = 0.0, zero_acc = 0.0, nonzero_acc = 0.0;
                int nonzero_count = 0, zero_count = 0;

                for (int s = 0; s < num; ++s) {
//                    qDebug() << "===Sample" << s;
                    for (int j = 0; j < label_size; ++j)
                    {
                        int index = s*label_size + j;
                        float dif = pow(target[index] - actual[index], 2);
                        acc += int(sqrt(dif) < acc_threshold);

                        zero_acc += (target[index] == 0.0 ? sqrt(dif) < acc_threshold : 0.0);
                        nonzero_acc += (target[index] != 0.0 ? sqrt(dif) < acc_threshold : 0.0);
                        zero_count += target[index] == 0.0;
                        nonzero_count += target[index] != 0.0;
//                        if (s<1)
//                            qDebug() << j << ":" << "t=" << (target[index]+1)*acc_threshold_value <<
//                                        ", a=" << (actual[index]+1)*acc_threshold_value <<
//                                        ", dif=" << dif*acc_threshold_value << ", acc=" << int(sqrt(dif) < acc_threshold);
                    }
                }

                test_acc += acc/float(label_size*num);
                test_zero_acc += zero_acc/float(zero_count);
                test_nonzero_acc += nonzero_acc/float(nonzero_count);

            }
            qDebug() << "Test phase, iteration" << solver->iter();
            float count = ceil(num_test_data/float(num));
            test_loss /= count;
            test_acc /= count;
            test_zero_acc /= count;
            test_nonzero_acc /= count;

            qDebug() << "   Test net: loss =" << test_loss <<  ", acc =" << test_acc << " (error:" << acc_threshold_value << "x" << acc_threshold_mul << ")"
                     << ", zero_acc =" << test_zero_acc << ", nonzero_acc =" << test_nonzero_acc << ", sum=" << test_zero_acc+test_nonzero_acc;
        }
#elif ATTRIBUTES_TEST_TYPE == AT_RECOLORAZING_SOFTMAX
        /// Test phase ///
        if (solver->iter() % test_interval == 0)
        {
            Caffe::set_mode(Caffe::GPU);
            boost::shared_ptr<Blob<float> > output_blob = test_net->blob_by_name("output");
            boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
            const int num = test_net->blob_by_name("data")->shape(0);
            const int label_size = output_blob->shape(1);
            float test_loss = 0.0;
            float test_acc = 0.0;

            for (unsigned int i = 0; i < num_test_data; i += num) {
                set_test_data_batch(i);
                float loss = 0.0;
                const std::vector<Blob<float>*>& result = test_net->Forward(&loss);   //forward
                test_acc += result[0]->cpu_data()[0];
                test_loss += loss;

            }
            qDebug() << "Test phase, iteration" << solver->iter();
            float count = ceil(num_test_data/float(num));
            test_loss /= count;
            test_acc /= count;
            qDebug() << "   Test net: loss =" << test_loss <<  ", acc =" << test_acc;


            //DEBUG:
            {
                boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
                const float* data = data_blob->cpu_data();
                const int batch_size = test_net->blob_by_name("data")->shape(0);
                const int sample_channels = test_net->blob_by_name("data")->shape(1);
                const int sample_height = test_net->blob_by_name("data")->shape(2);
                const int sample_width = test_net->blob_by_name("data")->shape(3);
                const int sample_size = sample_channels*sample_height*sample_width;

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
                }
            }
        }
#endif

        /// Treain phase ///
        Caffe::set_mode(Caffe::GPU);
        set_data_batch();
        solver->Step(1);

//        //DEBUG:
//        {
//            auto net = solver->net();
//            boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
//            const float* data = data_blob->cpu_data();
//            const int batch_size = net->blob_by_name("data")->shape(0);
//            const int sample_channels = net->blob_by_name("data")->shape(1);
//            const int sample_height = net->blob_by_name("data")->shape(2);
//            const int sample_width = net->blob_by_name("data")->shape(3);
//            const int sample_size = sample_channels*sample_height*sample_width;

//            for (int j = 0; j < batch_size; ++j) {
//                //Grayscale:
////                cv::Mat out_img = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j);

//                //Rgb:
//                cv::Mat out_img;
//                std::vector<cv::Mat> channels(3);
//                channels[0] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j);
//                channels[1] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width);
//                channels[2] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width*2);
//                cv::merge(channels, out_img);

//                out_img /= 0.00390625;
//                QString path = QString("/home/darkalert/Desktop/test/%1_s.png").arg(j);
//                cv::imwrite(path.toStdString(), out_img);
//            }
//        }


//        //DEBUG:
//        {
//            auto net = solver->net();
//            boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
//            boost::shared_ptr<Blob<float> > label_blob = net->blob_by_name("label");
//            const float* data = data_blob->cpu_data();
//            const float* labels = label_blob->cpu_data();
//            const int batch_size = net->blob_by_name("data")->shape(0);
//            const int sample_channels = net->blob_by_name("data")->shape(1);
//            const int sample_height = net->blob_by_name("data")->shape(2);
//            const int sample_width = net->blob_by_name("data")->shape(3);
//            const int sample_size = sample_channels*sample_height*sample_width;
//            const int label_channels = net->blob_by_name("label")->shape(1);
//            const int label_height = net->blob_by_name("label")->shape(2);
//            const int label_width = net->blob_by_name("label")->shape(3);
//            const int label_size = label_channels*label_height*label_width;

//            for (int j = 0; j < batch_size; ++j) {
//                cv::Mat out_img;
//                std::vector<cv::Mat> channels(3);
//                channels[0] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j);
//                channels[1] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width);
//                channels[2] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width*2);
//                cv::merge(channels, out_img);
//                out_img /= 0.00390625;
//                QString path = QString("/home/darkalert/Desktop/test/%1_s.png").arg(j);
//                cv::imwrite(path.toStdString(), out_img);
//            }
//        }
        //DEBUG:
//        {
//            auto net = solver->net();
//            boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
//            const float* data = data_blob->cpu_data();
//            const int batch_size = net->blob_by_name("data")->shape(0);
//            const int sample_channels = net->blob_by_name("data")->shape(1);
//            const int sample_height = net->blob_by_name("data")->shape(2);
//            const int sample_width = net->blob_by_name("data")->shape(3);
//            const int sample_size = sample_channels*sample_height*sample_width;

//            for (int j = 0; j < batch_size; ++j) {
//                cv::Mat out_img;
//                std::vector<cv::Mat> channels(3);
//                channels[0] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j);
//                channels[1] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width);
//                channels[2] = cv::Mat(sample_height,sample_width, CV_32FC1, const_cast<float*>(data) + sample_size*j + sample_height*sample_width*2);
//                cv::merge(channels, out_img);
//                out_img /= 0.00390625;
//                QString path = QString("/home/darkalert/Desktop/test/%1_s.png").arg(j);
//                cv::imwrite(path.toStdString(), out_img);
//            }
//        }
    }

    train_log.close();
    train_worker_is_running = false;
}

void AttributePredictorTrainer::train(const std::string &path_to_solver)
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
    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/classifier_hairs/best/tiny_gray_v3_r_iter_65000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/gena2017/backend/Resources/Nets/orig_x2_iter_55000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/resnet_imagenet/ResNet-50-model.caffemodel");

    //Log:
    qDebug() << "ClassifierTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "ClassifierTrainer: GPU mode has been set." : "ClassifierTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&AttributePredictorTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&AttributePredictorTrainer::train_by_worker, this);
    train_worker.detach();
}

void AttributePredictorTrainer::restore(const std::string &path_to_solver, const std::string &path_to_solverstate)
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
    prefetch_worker = std::thread(&AttributePredictorTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&AttributePredictorTrainer::train_by_worker, this);
    train_worker.detach();
}

void AttributePredictorTrainer::stop()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        helper::little_sleep(std::chrono::microseconds(100));
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

void AttributePredictorTrainer::pause()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        helper::little_sleep(std::chrono::microseconds(100));
    }

    qDebug() << "ClassifierTrainer: Training is paused. Current iter:" << solver->iter();
}

void AttributePredictorTrainer::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&AttributePredictorTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&AttributePredictorTrainer::train_by_worker, this);
    train_worker.detach();

    qDebug() << "ClassifierTrainer: Training is resumed.";
}
