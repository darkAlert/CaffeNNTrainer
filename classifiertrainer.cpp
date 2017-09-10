#include "classifiertrainer.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include <QFileInfo>
#include <QDebug>
#include <QTime>
#include <QDir>

#include "caffe/net.hpp"
#include "helper.h"
#include <cuda.h>


using namespace caffe;


ClassifierTrainer::ClassifierTrainer()
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

ClassifierTrainer::~ClassifierTrainer()
{
    stop();
}

void ClassifierTrainer::prefetch_batches_by_worker(unsigned int batch_margin)
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
        const cv::Size2i max_crop_offset = cv::Size2i(144,144);

        const bool mirror = false;

        const bool scale = true;
        const float max_scale_ratio = 0.2;

        const bool resize = false;
        const float max_resize_ratio = 0.5;
    } DISTORTS;

    //Distributions:
    boost::random::uniform_real_distribution<float> die_scale(0.0, DISTORTS.max_scale_ratio);
    boost::random::uniform_real_distribution<float> die_resize(0.0, DISTORTS.max_resize_ratio);
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
                float label_f;
                auto raw_sample = data_provider->get_train_sample(&label_f);
                cv::Mat sample_float;
                cv::Mat(raw_sample.rows, raw_sample.cols, CV_8UC1, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, CV_32FC1, 0.00390625);

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
                //Resize:
                if (DISTORTS.resize) {
                    float ratio = 1.0 - die_resize(rng);
                    cv::resize(sample_float,sample_float,cv::Size(0, 0), ratio, ratio, CV_INTER_AREA);
                    cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, CV_INTER_LINEAR);
                }
                //Mirror:
                if (DISTORTS.mirror && rand() < RAND_MAX/2) {
                    cv::flip(sample_float,sample_float,1);
                }
                //Resize to fit:
                if (sample_float.rows != height || sample_float.cols != width) {
                    auto inter = sample_float.cols > width ? CV_INTER_AREA : CV_INTER_LINEAR;
                    cv::resize(sample_float,sample_float,cv::Size(width, height), 0, 0, inter);
                }

//                sample_float = sample_float*255.0 - 127.5;
//                sample_float = sample_float*2.0 - 1.0;


                //Set data to the main triplet net:
                memcpy(&(data_batch.get()[i*sample_size]), sample_float.data, sizeof(float)*(sample_size));
                memcpy(&(labels_batch.get()[i*label_size]), &label_f, sizeof(float));
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

void ClassifierTrainer::prepare_test_data()
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
        float label_f;
        auto raw_sample = data_provider->get_test_sample(i, &label_f);
        cv::Mat sample_float;
        cv::Mat(raw_sample.rows, raw_sample.cols, CV_8UC1, const_cast<unsigned char*>(raw_sample.data)).convertTo(sample_float, CV_32FC1, 0.00390625);
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

        sample_float = sample_float*255.0 - 127.5;
//        sample_float = sample_float*2.0 - 1.0;

        //Set data to the main triplet net:
        memcpy(&(test_data.first.get())[i*sample_size], sample_float.data, sizeof(float)*sample_size);
        memcpy(&(test_data.second.get()[i*label_size]), &label_f, sizeof(float)*label_size);
    }

    data_provider->clear_test_batch();
}

void ClassifierTrainer::set_data_batch()
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
                qDebug() << "Waiting for data...";
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

void ClassifierTrainer::set_test_data_batch(int sample_index)
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
    auto rest = data_provider->total_test_samples()-sample_index >= num ? num : data_provider->total_test_samples()-sample_index;
    memcpy(data, &(test_data.first.get()[sample_index*sample_size]), sizeof(float)*sample_size*rest);
    memcpy(labels, &(test_data.second.get()[sample_index*label_size]), sizeof(float)*label_size*rest);
}

void ClassifierTrainer::openTrainData(const std::string &path_to_train)
{
    //Open data:
    if (data_provider == nullptr) {
        data_provider = std::make_shared<DataProviderSamples>(train_batch_size, data_provider_worker_count);
    }
    if (data_provider->open(path_to_train, test_samples_per_identity) == false) {
        qDebug() << "ClassifierTrainer: Unknown error while data loading. Training has been stopped.";
        return;
    }
}

void ClassifierTrainer::train_by_worker()
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
            Caffe::set_mode(Caffe::GPU);
            const int num = test_net->blob_by_name("data")->shape(0);
            float test_accuracy = 0.0;
            float test_loss = 0.0;
            for (unsigned int i = 0; i < num_test_data; i += num) {
                set_test_data_batch(i);
                float loss;
                const std::vector<Blob<float>*>& result = test_net->Forward(&loss);   //forward
                test_accuracy += result[0]->cpu_data()[0];
                test_loss += loss;
            }
            qDebug() << "Test phase, iteration" << solver->iter();
            test_accuracy /= ceil(num_test_data/num);
            test_loss /= ceil(num_test_data/num);
            qDebug() << "   Test net: accuracy =" << test_accuracy << ", loss =" << test_loss;
        }
        /// Treain phase ///
        Caffe::set_mode(Caffe::GPU);
        set_data_batch();
        solver->Step(1);
    }

    train_log.close();
    train_worker_is_running = false;
}

void ClassifierTrainer::train(const std::string &path_to_solver, const std::string &path_to_testnet)
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
    caffe::ReadNetParamsFromTextFileOrDie((path_to_testnet.empty() == false ? path_to_testnet : solver_param.net()), &test_net_param);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, test_net_batch_size);
    test_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(1)->set_dim(0, test_net_batch_size);
    test_net_param.mutable_state()->set_phase(Phase::TEST);
    test_net = std::make_shared<Net<float> >(test_net_param);
    solver->net().get()->ShareTrainedLayersWith(test_net.get());

    //Load weights:
    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/gena2017/backend/Resources/Nets/orig_x2_iter_55000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256/best/_iter_105000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet512maxout/best/_iter_130000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/tresnet256maxout/best/lfw_99p32/x2_ms_iter_6000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256maxout/best/ms_10575/ms10575_r0_iter_200000.caffemodel");

    //Log:
    qDebug() << "ClassifierTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "ClassifierTrainer: GPU mode has been set." : "ClassifierTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&ClassifierTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&ClassifierTrainer::train_by_worker, this);
    train_worker.detach();
}

void ClassifierTrainer::restore(const std::string &path_to_solver, const std::string &path_to_solverstate, const std::string &path_to_testnet)
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
    caffe::ReadNetParamsFromTextFileOrDie((path_to_testnet.empty() == false ? path_to_testnet : solver_param.net()), &test_net_param);
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
    prefetch_worker = std::thread(&ClassifierTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&ClassifierTrainer::train_by_worker, this);
    train_worker.detach();
}

void ClassifierTrainer::stop()
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

void ClassifierTrainer::pause()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        helper::little_sleep(std::chrono::microseconds(100));
    }

    qDebug() << "ClassifierTrainer: Training is paused. Current iter:" << solver->iter();
}

void ClassifierTrainer::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&ClassifierTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&ClassifierTrainer::train_by_worker, this);
    train_worker.detach();

    qDebug() << "ClassifierTrainer: Training is resumed.";
}
