#include "fctrainer.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include "caffe/net.hpp"
#include <cuda.h>

#include <QDebug>
#include <QTime>
#include <QFile>
#include <QFileInfo>

using namespace caffe;


#define LABEL_LANDMARKS_68P 0
#define LABEL_RECT_2P 1
const int LABEL_TYPE = LABEL_LANDMARKS_68P;

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


FCTrainer::FCTrainer()
{
    solver = nullptr;
    running = false;
    prefetch_worker_is_running = false;
    train_worker_is_running = false;
    test_net = nullptr;
    test_data.first = nullptr;
    test_data.second = nullptr;
}

FCTrainer::~FCTrainer()
{
    stop();
}

void FCTrainer::prefetch_batches_by_worker(unsigned int batch_margin)
{
    prefetch_worker_is_running = true;

    //Batch params:
    const int batch_size = solver->net()->blob_by_name("data")->shape(0);
    const int channels = solver->net()->blob_by_name("data")->shape(1);
    const int sample_size = channels;
    const int label_size = solver->net()->blob_by_name("label")->shape(1);

    //Distributions:
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
                const std::vector<float> raw_label = raw_train_samples[i].second;
                const std::vector<float> raw_sample = raw_train_samples[i].first;
                std::vector<float> sample(raw_sample.begin(), raw_sample.end());
                std::vector<float> label(raw_label.begin(), raw_label.end());

                //Set sample and label to the net:
                memcpy(&(data_batch.get()[i*sample_size]), sample.data(), sizeof(float)*(sample_size));
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

void FCTrainer::prepare_test_data()
{
    if (test_net == nullptr) return;

    //Data params:
    const int channels =test_net->blob_by_name("data")->shape(1);
    const int sample_size = channels;
    const int label_size = test_net->blob_by_name("label")->shape(1);

    //Allocate memory:
    test_data.first.reset(new float[num_test_samples*sample_size]);
    test_data.second.reset(new float[num_test_samples*label_size]);

    //Prepare data:
    for (unsigned int i = 0; i < num_test_samples; ++i)
    {
        const std::vector<float> raw_label = raw_train_samples[i].second;
        const std::vector<float> raw_sample = raw_train_samples[i].first;
        std::vector<float> sample(raw_sample.begin(), raw_sample.end());
        std::vector<float> label(raw_label.begin(), raw_label.end());

        //Set data and label to the net:
        memcpy(&(test_data.first.get())[i*sample_size], sample.data(), sizeof(float)*sample_size);
        memcpy(&(test_data.second.get()[i*label_size]), label.data(), sizeof(float)*label_size);
    }
}

void FCTrainer::set_data_batch()
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

void FCTrainer::set_test_data_batch(int sample_index)
{
    assert(test_net != nullptr);
    boost::shared_ptr<Blob<float> > data_blob = test_net->blob_by_name("data");
    boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");

    //Set data to the input of the net:
    const int num = data_blob->shape(0);
    assert((unsigned int)num == num_test_samples);
    const int sample_size = data_blob->shape(1);
    const int label_size = label_blob->shape(1);
    float* data = data_blob->mutable_cpu_data();
    float* labels = label_blob->mutable_cpu_data();

    //Set the data and labels:
    auto rest = num_test_samples-sample_index >= (unsigned int)num ? (unsigned int)num : num_test_samples-sample_index;
    memcpy(data, &(test_data.first.get()[sample_index*sample_size]), sizeof(float)*sample_size*rest);
    memcpy(labels, &(test_data.second.get()[sample_index*label_size]), sizeof(float)*label_size*rest);
}

bool FCTrainer::read_raw_samples(const std::string &path_to_csv)
{
    //Open csv containing dirs:
    std::ifstream csv_file(path_to_csv.c_str(), std::ifstream::in);
    if (!csv_file) {
        qDebug() << "FCTrainer: No valid input file was given, please check the given filename.";
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

        //Push sample and label:
        if (count < num_test_samples) {
            raw_test_samples.push_back(std::make_pair(sample, label));
        }
        else {
            raw_train_samples.push_back(std::make_pair(sample, label));
        }
        ++count;
    }
    assert(raw_test_samples.size() == num_test_samples);

    qDebug() << "FCTrainer: Sample have been loaded. Train sampels:" << raw_train_samples.size() << ", test samples:" << raw_test_samples.size();
    return true;
}

void FCTrainer::train_by_worker()
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
            const float acc_threshold = 0.010;   // in pixels
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

            //Set test data:
            set_test_data_batch(0);
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
            }
            test_x_acc += x_acc/float(label_size*0.5*num);
            test_y_acc += y_acc/float(label_size*0.5*num);
            test_points_acc += points_acc/float(label_size*0.5*num);
            test_contour_acc += contour_acc/float(CONTOUR_COUNT*num);
            test_eyebrows_acc += eyebrows_acc/float(EYEBROWS_COUNT*num);
            test_nose_acc += nose_acc/float(NOSE_COUNT*num);
            test_eyes_acc += eyes_acc/float(EYES_COUNT*num);
            test_mouth_acc += mouth_acc/float(MOUTH_COUNT*num);

            qDebug() << "Test phase, iteration" << solver->iter();
            float count = 1;
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

            //DEBUG:
            {
                const float* src_landmarks = data_blob->cpu_data();
                const float* dst_landmarks_target = label_blob->cpu_data();
                const float* dst_landmarks_actual = output_blob->cpu_data();
                const int line_size = 5;
                cv::Size out_size(550, 250);

                for (unsigned int j = 0; j < num_test_samples; ++j)
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

                    QString img_name = QString("/home/darkalert/Desktop/test/%1.png").arg(j);
                    cv::imwrite(img_name.toStdString(), out_img);
                }
            }
        }

        /// Treain phase ///
        Caffe::set_mode(Caffe::GPU);
        set_data_batch();
        solver->Step(1);
    }

    train_worker_is_running = false;
}

void FCTrainer::train(const std::string &path_to_solver)
{
    if (running == true) {
        qDebug() << "FCTrainer: Training is already running.";
        return;
    }

    if (raw_train_samples.empty()) {
        qDebug() << "FCTrainer: Training data are empty!";
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
    solver->net()->CopyTrainedLayersFrom("/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/best/p69as68v3.caffemodel");

    //Log:
    qDebug() << "FCTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "FCTrainer: GPU mode has been set." : "FCTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&FCTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&FCTrainer::train_by_worker, this);
    train_worker.detach();
}

void FCTrainer::restore(const std::string &path_to_solver, const std::string &path_to_solverstate)
{
    if (running == true) {
        qDebug() << "FCTrainer: Training is already running.";
        return;
    }

    if (raw_train_samples.empty()) {
        qDebug() << "FCTrainer: Training data are empty!";
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
    qDebug() << "FCTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << "FCTrainer: Solver's state has been restored from" << QString::fromStdString(path_to_solverstate);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "FCTrainer: GPU mode has been set." : "FCTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&FCTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&FCTrainer::train_by_worker, this);
    train_worker.detach();
}

void FCTrainer::stop()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
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

void FCTrainer::pause()
{
    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    qDebug() << "ClassifierTrainer: Training is paused. Current iter:" << solver->iter();
}

void FCTrainer::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&FCTrainer::prefetch_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&FCTrainer::train_by_worker, this);
    train_worker.detach();

    qDebug() << "FCTrainer: Training is resumed.";
}
