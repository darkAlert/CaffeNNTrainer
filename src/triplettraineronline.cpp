#include "triplettraineronline.h"

#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include <QFileInfo>
#include <QDebug>
#include <QTime>
#include <QDir>

#include "caffe/net.hpp"
#include <cuda.h>


using namespace caffe;


TripletTrainerOnline::TripletTrainerOnline()
{
    data_provider = nullptr;
    solver = nullptr;
    test_net = nullptr;
    dist_net = nullptr;
    running = false;
    prefetch_worker_is_running = false;
    train_worker_is_running = false;
}

TripletTrainerOnline::~TripletTrainerOnline()
{
    stop();
}

void TripletTrainerOnline::prefetch_data_batches_by_worker(unsigned int batch_margin)
{
    prefetch_worker_is_running = true;

    const int batch_size = solver->net()->blob_by_name("data")->shape(0);
    const int channels = solver->net()->blob_by_name("data")->shape(1);
    const int height = solver->net()->blob_by_name("data")->shape(2);
    const int width = solver->net()->blob_by_name("data")->shape(3);
    const int sample_size = channels*height*width;
    const int label_size = solver->net()->blob_by_name("label")->shape(1) * solver->net()->blob_by_name("label")->shape(2) * solver->net()->blob_by_name("label")->shape(3);
    const unsigned int train_data_size = prefetch_identities_in_memory > 0 ? prefetch_identities_in_memory : identities_in_memory;

    assert(TRIPLET_PARAMS.triplet_batch_size == batch_size);

    struct Distortions {
        const bool crop = true;
        const cv::Size2i max_crop_offset = cv::Size2i(144,144);

        const bool mirror = true;

        const bool scale = true;
        const float max_scale_ratio = 0.1;

        const bool resize = false;
        const float max_resize_ratio = 0.5;

        const float resize_negatives_prob = -1.0;           //probability
        const float resize_negatives_ratio[2] = {0.05, 0.2}; //min, max
    } DISTORTS;

    srand(std::time(nullptr));
    rng.seed(std::time(nullptr));
    boost::random::uniform_real_distribution<float> die_one(0.0, 1.0);
    boost::random::uniform_real_distribution<float> die_f(0.0, DISTORTS.max_scale_ratio);
    boost::random::uniform_real_distribution<float> die_resize(0.0, DISTORTS.max_resize_ratio);
    boost::random::uniform_int_distribution<int> die_tc(0, identities_in_memory-1);
    boost::random::uniform_real_distribution<float> die_resize_negatives(DISTORTS.resize_negatives_ratio[0], DISTORTS.resize_negatives_ratio[1]);

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

            /// Pick anchor and positive samples for the main triplet net ///
            for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i)
            {
                if (current_train_data_i >= train_data_size) {
                    auto updated = data_provider->update();
                    current_train_data_i = 0;
//                    qDebug() << "TripletTrainerOnline:" << updated << "identities have been updated.";
                }
                float anchor_label_f;
                auto anchor_identity_samples = data_provider->identity_samples(current_train_data_i, &anchor_label_f);
                ++current_train_data_i;

                /// Select and prepare an anchor sample ///
                unsigned int anchor_sample_index = rand()%anchor_identity_samples.size();
                cv::Mat raw_sample = anchor_identity_samples[anchor_sample_index];
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
                    float ratio = 1.0 + die_f(rng);
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
//                const cv::Size2i max_size(width*0.5, height*0.5);
//                //Black rect:
//                if (DISTORTS.black_box) {
//                    cv::Rect black_rect;
//                    black_rect.width = rand()%max_size.width;
//                    black_rect.height = rand()%max_size.height;
//                    black_rect.x = rand()%(width - black_rect.width +1);
//                    black_rect.y = rand()%(height - black_rect.height +1);
//                    cv::rectangle(sample_float, black_rect, 0.0, CV_FILLED);
//                }

                //Set data to the main triplet net:
                memcpy(&(data_batch.get()[i*3*sample_size]), sample_float.data, sizeof(float)*(sample_size)); //value 3 due to anchor+pos+neg
                memcpy(&(labels_batch.get()[i*3*label_size]), &anchor_label_f, sizeof(float));

                /// Select a positive sample ///
                unsigned int pos_sample_index;
                int count = 0;
                do {
                    pos_sample_index = rand()%anchor_identity_samples.size();
                    ++count;
                } while (pos_sample_index == anchor_sample_index && count < 5);

                auto raw_pos_sample = anchor_identity_samples[pos_sample_index];
                cv::Mat(raw_pos_sample.rows, raw_pos_sample.cols, CV_8UC1, const_cast<unsigned char*>(raw_pos_sample.data)).convertTo(sample_float, CV_32FC1, 0.00390625);

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
                    float ratio = 1.0 + die_f(rng);
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

                //Set data:
                memcpy(&(data_batch.get()[(i*3+1)*sample_size]), sample_float.data, sizeof(float)*(sample_size));
                memcpy(&(labels_batch.get()[(i*3+1)*label_size]), &anchor_label_f, sizeof(float));
            }

            //Push a prepared data batch (for the main triplet net) to the prefetched_data:
            mutex_prefetch_data.lock();
            prefetched_data.push_back(std::make_pair(data_batch,labels_batch));
            mutex_prefetch_data.unlock();

            /// Pick remained negatives samples for the represent net ///
            for (int k = 0; k < TRIPLET_PARAMS.represent_mult; ++k)
            {
                //Auxiliary represent net:
                std::shared_ptr<float> represent_data_batch;
                std::shared_ptr<float> represent_labels_batch;
                represent_data_batch.reset(new float[batch_size*sample_size]);
                represent_labels_batch.reset(new float[batch_size*label_size]);

                //First copy anchor samples to the beginning of represent batch:
                if (k == 0) {
                    for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i) {
                        memcpy(&(represent_data_batch.get()[i*sample_size]), &(data_batch.get()[i*3*sample_size]), sizeof(float)*sample_size);  //anchor
                        memcpy(&(represent_labels_batch.get()[i*label_size]), &(labels_batch.get()[i*3*label_size]), sizeof(float)*label_size);
                        memcpy(&(represent_data_batch.get()[(i+TRIPLET_PARAMS.triplets_num)*sample_size]), &(data_batch.get()[(i*3+1)*sample_size]), sizeof(float)*sample_size);  //positive
                        memcpy(&(represent_labels_batch.get()[(i+TRIPLET_PARAMS.triplets_num)*label_size]), &(labels_batch.get()[(i*3+1)*label_size]), sizeof(float)*label_size);
                    }
                }

                for (int i = (k == 0 ? TRIPLET_PARAMS.triplets_num : 0); i < batch_size; ++i)
                {
                    unsigned int neg_i = die_tc(rng);
                    float neg_label_f;
                    auto neg_identity_samples = data_provider->identity_samples(neg_i, &neg_label_f);
                    cv::Mat raw_neg_sample = neg_identity_samples[rand()%neg_identity_samples.size()];
                    cv::Mat sample_float;
                    cv::Mat(raw_neg_sample.rows, raw_neg_sample.cols, CV_8UC1, const_cast<unsigned char*>(raw_neg_sample.data)).convertTo(sample_float, CV_32FC1, 0.00390625);

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
                        float ratio = 1.0 + die_f(rng);
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
                    //Resize negatives:
                    if (DISTORTS.resize_negatives_prob > 0.0 && die_one(rng) < DISTORTS.resize_negatives_prob) {
                        float ratio = die_resize_negatives(rng);
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

                    //Set data:
                    memcpy(&(represent_data_batch.get()[i*sample_size]), sample_float.data, sizeof(float)*(sample_size));
                    memcpy(&(represent_labels_batch.get()[i*label_size]), &neg_label_f, sizeof(float)*label_size);
                }

                //Push a prepared data batch (for the main triplet net) to the prefetched_data:
                mutex_prefetch_data.lock();
                prefetched_represent_data.push_back(std::make_pair(represent_data_batch, represent_labels_batch));
                mutex_prefetch_data.unlock();
            }
        }
        else {
            mutex_prefetch_data.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(25000));
        }
    }

    prefetch_worker_is_running = false;
}

std::shared_ptr<float> TripletTrainerOnline::set_represent_data_batch()
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

    assert(num == TRIPLET_PARAMS.triplet_batch_size);
    bool waiting_flag = false;

    bool flag;
    do {
        //Check an availability of data:
        mutex_prefetch_data.lock();
        flag = prefetched_represent_data.empty();
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
        auto represent_data_batch = prefetched_represent_data.front();
        prefetched_represent_data.pop_front();
        mutex_prefetch_data.unlock();

        //Set the data and labels:
        memcpy(data, represent_data_batch.first.get(), sizeof(float)*sample_size*num);
        memcpy(labels, represent_data_batch.second.get(), sizeof(float)*label_size*num);

        return represent_data_batch.first;

    } while(flag);

    return nullptr;
}

void TripletTrainerOnline::openTrainData(const std::string &path_to_train)
{
    //Open IDX:
    if (data_provider == nullptr) {
        data_provider = std::make_shared<DataProviderIdentities>(identities_in_memory, prefetch_identities_in_memory, data_provider_worker_count);
    }
    if (data_provider->open(path_to_train) == false) {
        qDebug() << "TripletTrainerOnline: Unknown error while data loading. Training has been stopped.";
        return;
    }

    current_train_data_i = 0;
}

void TripletTrainerOnline::train_by_worker()
{
    train_worker_is_running = true;

    //Solver's params:
    auto param = solver->param();
    const int32_t test_iter = 423*5;//param.test_iter().data()[0];
    const int32_t test_interval = param.test_interval();

    //Other params:
    boost::shared_ptr<Blob<float> > vec_blob = dist_net->blob_by_name("data");
    const int feat_size = vec_blob->shape(3);

    assert(solver->net()->blob_by_name("data")->shape(0) == TRIPLET_PARAMS.triplet_batch_size);
    assert(solver->net()->blob_by_name("label")->shape(0) == TRIPLET_PARAMS.triplet_batch_size);
    assert(dist_net->blob_by_name("data")->shape(0) == TRIPLET_PARAMS.dist_batch_size);
    assert(solver->net()->blob_by_name("feat")->shape(1) == feat_size);

    //Set mode:
    if (param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
//        Caffe::DeviceQuery();
        Caffe::SetDevice(0);
        Caffe::set_solver_count(1);
        Caffe::set_mode(Caffe::GPU);
    }
    else {
        Caffe::set_mode(Caffe::CPU);
    }

    //Logger:
    std::ofstream train_log(param.snapshot_prefix() + "train_log.txt", std::ios::app);

    //Memory:
    int label_size = 1;
    std::shared_ptr<float> anchor_feats, positive_feats, negative_feats;
    std::shared_ptr<float> anchor_labels, positive_labels, negative_labels;
    anchor_feats.reset(new float[TRIPLET_PARAMS.triplets_num*feat_size]);
    anchor_labels.reset(new float[TRIPLET_PARAMS.triplets_num*label_size]);
    positive_feats.reset(new float[TRIPLET_PARAMS.triplets_num*feat_size]);
    positive_labels.reset(new float[TRIPLET_PARAMS.triplets_num*label_size]);
    negative_feats.reset(new float[TRIPLET_PARAMS.negs_num*feat_size]);
    negative_labels.reset(new float[TRIPLET_PARAMS.negs_num*label_size]);
    std::shared_ptr<float> dist_net_input;
    dist_net_input.reset(new float[TRIPLET_PARAMS.dist_batch_size*feat_size*2]);   //value 2 due to a pairs
    std::shared_ptr<float> negatives_template;
    negatives_template.reset(new float[TRIPLET_PARAMS.negs_num*feat_size*2]);   //negatives + empty place for anchors

    double mean_bad_triplets = 0.0;
    double mean_good_neg_per_triplet = 0.0;
    double mean_bad_triplet_diff = 0.0;
    double mean_positive_dist= 0.0;

    qDebug() << "prefetch_identities_in_memory=" << prefetch_identities_in_memory;
    qDebug() << "identities_in_memory=" << identities_in_memory;

    while(running)
    {
        /// Test phase ///
        if (solver->iter() % test_interval == 0) {
            float test_accuracy = 0.0;
            float total_loss = 0.0;
            float test_pos_mean_dist = 0.0, test_neg_mean_dist = 0.0, test_dist_ratio = 0.0;
            int total_pos_pairs = 0, total_neg_pairs = 0;
/*
            for (int i = 0; i < test_iter; ++i) {
                float loss;
                const std::vector<Blob<float>*>& result = test_net->Forward(&loss);
//                test_accuracy += result[0]->cpu_data()[0];
                total_loss += loss;

                //Calculate accuracy:
                boost::shared_ptr<Blob<float> > output_blob = test_net->blob_by_name("sum");
                float* output = output_blob->mutable_cpu_data();
                boost::shared_ptr<Blob<float> > label_blob = test_net->blob_by_name("label");
                float* label = label_blob->mutable_cpu_data();
                for (int j = 0; j < 5; ++j) {
                    if (label[j] > 0.5) {
                        test_accuracy += output[j] < 1.0 ? 1.0 : 0.0;
                        test_pos_mean_dist += output[j];
                        ++total_pos_pairs;
                    }
                    else {
                        test_accuracy += output[j] >= 1.0 ? 1.0 : 0.0;
                        test_neg_mean_dist += output[j];
                        ++total_neg_pairs;
                    }
//                    if (i == 0)
//                    qDebug() << "label=" << label[j] <<", output=" << output[j];
                }
            }
            test_pos_mean_dist /= float(total_pos_pairs);
            test_neg_mean_dist /= float(total_neg_pairs);
            test_dist_ratio = test_pos_mean_dist / test_neg_mean_dist;
            test_accuracy /= 5.0;
            qDebug() << qPrintable(QString("[%1] Iter %2, test accuracy: %3, test loss: %4")
                                   .arg(QTime::currentTime().toString("h:mm:ss"))
                                   .arg(solver->iter())
                                   .arg(test_accuracy/test_iter)
                                   .arg(total_loss/test_iter));
            train_log << solver->iter() << " " << test_accuracy/test_iter << " " << total_loss/test_iter << "\n";
            train_log.flush();
*/
            //Mean:
//            qDebug() << "TEST: mean pos distance =" <<  test_pos_mean_dist << ", mean neg distance =" << test_neg_mean_dist << ", ratio =" << test_dist_ratio;
            qDebug() << "mean_positive_dist:" <<  mean_positive_dist;
            qDebug() << "Train: mean goods per triplet:" <<  mean_good_neg_per_triplet
                     << ", mean bads:" <<  mean_bad_triplets
                     << ", mean diff in bads:" << mean_bad_triplet_diff;
            mean_bad_triplets = 0.0;
            mean_good_neg_per_triplet = 0.0;
            mean_bad_triplet_diff = 0.0;
            mean_positive_dist = 0.0;
        }

        /// Phase 1: representing samples ///
        std::vector<std::shared_ptr<float> > represent_data;
        auto net = solver->net();

        for (int k = 0; k < TRIPLET_PARAMS.represent_mult; ++k)
        {
            //Forward:
            represent_data.push_back(set_represent_data_batch());
            Caffe::set_mode(Caffe::GPU);
//            net->layer_by_name("drop1")->set_phase(Phase::TEST);
            net->Forward();
//            //DEBUG:
//            {
//                boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
//                float* data = data_blob->mutable_cpu_data();
//                boost::shared_ptr<Blob<float> > label_blob = net->blob_by_name("label");
//                float* label = label_blob->mutable_cpu_data();
//                int dir = solver->iter();
//                if (QDir(QString("/home/vitvit/Desktop/temp/%1/").arg(dir)).exists() == false) {
//                    QDir().mkdir(QString("/home/vitvit/Desktop/temp/%1/").arg(dir));
//                }
//                for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i) {
//                    cv::Mat temp = cv::Mat(128,128,CV_32FC1,data+i*3*128*128);
//                    temp = temp/0.00390625;
//                    QString img_name = QString("/home/vitvit/Desktop/temp/%1/t%2_a_L%3.png").arg(dir).arg(i).arg(uint32_t(label[i*3]));
//                    cv::imwrite(img_name.toStdString(), temp);

//                    temp = cv::Mat(128,128,CV_32FC1,data+(i*3+1)*128*128);
//                    temp = temp/0.00390625;
//                    img_name = QString("/home/vitvit/Desktop/temp/%1/t%2_p_L%3.png").arg(dir).arg(i).arg(uint32_t(label[i*3+1]));
//                    cv::imwrite(img_name.toStdString(), temp);

//                    temp = cv::Mat(128,128,CV_32FC1,data+(i*3+2)*128*128);
//                    temp = temp/0.00390625;
//                    img_name = QString("/home/vitvit/Desktop/temp/%1/t%2_n_L%3.png").arg(dir).arg(i).arg(uint32_t(label[i*3+2]));
//                    cv::imwrite(img_name.toStdString(), temp);
//                }
//                qDebug() << "done.";
//                return;
//            }

            boost::shared_ptr<Blob<float> > feats_blob =net->blob_by_name("feat");
            boost::shared_ptr<Blob<float> > labels_blob =net->blob_by_name("label");
            const float* feats_data = feats_blob->cpu_data();
            const float* labels_data = labels_blob->cpu_data();

            //Take anchor and positive feats and labels:
            if (k == 0) {
                memcpy(anchor_feats.get(), feats_data, sizeof(float)*(TRIPLET_PARAMS.triplets_num*feat_size));
                memcpy(anchor_labels.get(), labels_data, sizeof(float)*(TRIPLET_PARAMS.triplets_num*label_size));
                memcpy(positive_feats.get(), feats_data+(TRIPLET_PARAMS.triplets_num*feat_size), sizeof(float)*(TRIPLET_PARAMS.triplets_num*feat_size));
                memcpy(positive_labels.get(), labels_data+(TRIPLET_PARAMS.triplets_num*label_size), sizeof(float)*(TRIPLET_PARAMS.triplets_num*label_size));
            }
            //Take negative feats and labels:
            if (k == 0) {
                int dest_index = 0;
                int source_index = TRIPLET_PARAMS.triplets_num*2;    //mult by 2 due to (anchor + positive)
                int feat_count = TRIPLET_PARAMS.triplet_batch_size - TRIPLET_PARAMS.triplets_num*2;
                memcpy(&(negative_feats.get()[dest_index*feat_size]), &(feats_data[source_index*feat_size]), sizeof(float)*(feat_count*feat_size));
                memcpy(&(negative_labels.get()[dest_index*label_size]), &(labels_data[source_index*label_size]), sizeof(float)*(feat_count*label_size));
            }
            else {
                int dest_index = (TRIPLET_PARAMS.triplet_batch_size - TRIPLET_PARAMS.triplets_num*2) + (k-1)*(TRIPLET_PARAMS.triplet_batch_size);
                int source_index = 0;
                int feat_count = TRIPLET_PARAMS.triplet_batch_size;
                memcpy(&(negative_feats.get()[dest_index*feat_size]), &(feats_data[source_index*feat_size]), sizeof(float)*(feat_count*feat_size));
                memcpy(&(negative_labels.get()[dest_index*label_size]), &(labels_data[source_index*label_size]), sizeof(float)*(feat_count*label_size));
            }
        }
        assert(((TRIPLET_PARAMS.triplet_batch_size - TRIPLET_PARAMS.triplets_num*2) + (TRIPLET_PARAMS.represent_mult-1)*(TRIPLET_PARAMS.triplet_batch_size)) == TRIPLET_PARAMS.negs_num);


        /// Phase 2: computing the distances ///
        //Make positive pairs (among whom we compute the distances):
        for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i) {
            memcpy(&(dist_net_input.get()[i*2*feat_size]), &(anchor_feats.get()[i*feat_size]), sizeof(float)*(feat_size));
            memcpy(&(dist_net_input.get()[i*2*feat_size+feat_size]), &(positive_feats.get()[i*feat_size]), sizeof(float)*(feat_size));
        }

        //Make negatives template:
        for (int j = 0; j < TRIPLET_PARAMS.negs_num; ++j) {
            memcpy(&(negatives_template.get()[(j*2+1)*feat_size]), &(negative_feats.get()[j*feat_size]), sizeof(float)*(feat_size));
        }

        //Make negative pairs (among whom we compute the distances):
        for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i)
        {
            //Copy negatives template into dest (distance net):
            int dest_index = (TRIPLET_PARAMS.triplets_num + i*TRIPLET_PARAMS.negs_num)*2;
            memcpy(&(dist_net_input.get()[dest_index*feat_size]), negatives_template.get(), sizeof(float)*TRIPLET_PARAMS.negs_num*2*feat_size);

            //Copy positives:
            for (int j = 0; j < TRIPLET_PARAMS.negs_num; ++j) {
                memcpy(&(dist_net_input.get()[dest_index*feat_size]), &(anchor_feats.get()[i*feat_size]), sizeof(float)*(feat_size));
                dest_index += 2;
            }
        }

        //Set data to the input of the dist net:
        boost::shared_ptr<Blob<float> > input_blob = dist_net->blob_by_name("data");
        const int input_size = input_blob->shape(0) * input_blob->shape(3) * input_blob->shape(2) * input_blob->shape(1);
        assert(input_size == TRIPLET_PARAMS.dist_batch_size*feat_size*2);
        float* input_data = input_blob->mutable_cpu_data();
        memcpy(input_data, dist_net_input.get(), sizeof(float)*input_size);

        //Forward:
        Caffe::set_mode(Caffe::GPU);
        dist_net->Forward();
        boost::shared_ptr<Blob<float> > dist_blob = dist_net->blob_by_name("output");
        const float* dist_data = dist_blob->cpu_data();
        float* anchor_labels_raw = anchor_labels.get();
        float* negative_labels_raw = negative_labels.get();

        //Pick semi-hard negatives for each triplet:
        std::vector<std::vector<int> > triplet_neg_indexes;
        triplet_neg_indexes.resize(TRIPLET_PARAMS.triplets_num);
        int bad_triplets = 0;
        double bad_diff = 0.0;
        double positive_dist = 0.0;

        for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i)
        {
            int offset = TRIPLET_PARAMS.triplets_num + i*(TRIPLET_PARAMS.negs_num);

            /// Strategy 2: Semi-hard triplets ///
            float min_dist = 999.0;
            int min_index = 0;
            for (int j = 0; j < TRIPLET_PARAMS.negs_num; ++j) {
                if ((anchor_labels_raw[i] != negative_labels_raw[j])) {
                    if (dist_data[offset+j] < min_dist) {
                        min_dist = dist_data[offset+j];
                        min_index = j;
                    }
                    if (dist_data[i] - dist_data[offset+j] + TRIPLET_PARAMS.margin > 0) {
                        ++bad_triplets;
                    }
                }
            }
            triplet_neg_indexes[i].push_back(min_index);
            bad_diff += (dist_data[i] - min_dist + TRIPLET_PARAMS.margin);
            positive_dist += dist_data[i];
        }
        mean_bad_triplets += (float(bad_triplets) / float(TRIPLET_PARAMS.triplets_num) / float(test_interval));   //s3, s4, s5
        bad_triplets = 0; //for s2 only
        mean_bad_triplet_diff += (bad_diff / float(TRIPLET_PARAMS.triplets_num-bad_triplets) / float(test_interval));    //s2, s3, s4, s5
        mean_positive_dist += (positive_dist / float(TRIPLET_PARAMS.triplets_num) / float(test_interval));


        /// Phase 3: training the triplet net ///
        boost::shared_ptr<Blob<float> > data_blob = solver->net()->blob_by_name("data");
        boost::shared_ptr<Blob<float> > label_blob = solver->net()->blob_by_name("label");
        const int num = data_blob->shape(0);
        const int sample_size = data_blob->shape(1)*data_blob->shape(2)*data_blob->shape(3);

        //Take data batch (for the main triplet net):
        mutex_prefetch_data.lock();
        auto triplet_data_batch = prefetched_data.front();
        prefetched_data.pop_front();
        mutex_prefetch_data.unlock();

        //Add negatives to a triplet set:
        for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i)
        {
            int j = rand()%triplet_neg_indexes[i].size();    //random negative from suitable ones
            int dest_index = i*3+2;                          //value 2 due to anchor+positive
            int batch_index = (triplet_neg_indexes[i][j] + TRIPLET_PARAMS.triplets_num*2) / TRIPLET_PARAMS.triplet_batch_size;
            int source_index = (triplet_neg_indexes[i][j] + TRIPLET_PARAMS.triplets_num*2) % TRIPLET_PARAMS.triplet_batch_size;

            memcpy(&(triplet_data_batch.first.get()[dest_index*sample_size]), &(represent_data[batch_index].get()[source_index*sample_size]), sizeof(float)*(sample_size));
            memcpy(&(triplet_data_batch.second.get()[dest_index*label_size]), &(negative_labels.get()[triplet_neg_indexes[i][j]*label_size]), sizeof(float)*(label_size));
        }

        //Set triplet data to the main triplet net:
        float* data = data_blob->mutable_cpu_data();
        float* labels = label_blob->mutable_cpu_data();
        memcpy(data, triplet_data_batch.first.get(), sizeof(float)*sample_size*num);
        memcpy(labels, triplet_data_batch.second.get(), sizeof(float)*label_size*num);

        //Forward and backward:
        Caffe::set_mode(Caffe::GPU);
//        solver->net()->layer_by_name("drop1")->set_phase(Phase::TRAIN);
        solver->Step(1);
//        //DEBUG:
//        {
//            boost::shared_ptr<Blob<float> > data_blob = net->blob_by_name("data");
//            float* data = data_blob->mutable_cpu_data();
//            boost::shared_ptr<Blob<float> > label_blob = net->blob_by_name("label");
//            float* label = label_blob->mutable_cpu_data();
//            int dir = solver->iter();
//            if (QDir(QString("/home/vitvit/Desktop/temp/%1/").arg(dir)).exists() == false) {
//                QDir().mkdir(QString("/home/vitvit/Desktop/temp/%1/").arg(dir));
//            }
//            for (int i = 0; i < TRIPLET_PARAMS.triplets_num; ++i) {
//                cv::Mat temp = cv::Mat(128,128,CV_32FC1,data+i*3*128*128);
//                temp = temp/0.00390625;
//                QString img_name = QString("/home/vitvit/Desktop/temp/%1/t%2_a_L%3.png").arg(dir).arg(i).arg(uint32_t(label[i*3]));
//                cv::imwrite(img_name.toStdString(), temp);

//                temp = cv::Mat(128,128,CV_32FC1,data+(i*3+1)*128*128);
//                temp = temp/0.00390625;
//                img_name = QString("/home/vitvit/Desktop/temp/%1/t%2_p_L%3.png").arg(dir).arg(i).arg(uint32_t(label[i*3+1]));
//                cv::imwrite(img_name.toStdString(), temp);

//                temp = cv::Mat(128,128,CV_32FC1,data+(i*3+2)*128*128);
//                temp = temp/0.00390625;
//                img_name = QString("/home/vitvit/Desktop/temp/%1/t%2_n_L%3.png").arg(dir).arg(i).arg(uint32_t(label[i*3+2]));
//                cv::imwrite(img_name.toStdString(), temp);
//            }
////            qDebug() << "done.";
////            return;
//        }
    }

    train_log.close();
    train_worker_is_running = false;
}


void TripletTrainerOnline::train(const std::string &path_to_solver)
{
    if (running == true) {
        qDebug() << "NNTrainer: Training is already running.";
        return;
    }

    if (data_provider->total_identities() == 0) {
        qDebug() << "NNTrainer: Training data are empty!";
    }

    //Load solver:
    Caffe::set_mode(Caffe::GPU);
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(path_to_solver, &solver_param);
    solver_param.set_test_initialization(false);
//    auto net_param = solver_param.mutable_net_param();
//    net_param->mutable_layer(0)->mutable_data_param()->set_batch_size(TRIPLET_PARAMS.anchor_num*(TRIPLET_PARAMS.pos_per_anchor+TRIPLET_PARAMS.neg_per_anchor+1));
    solver = std::make_shared<SGDSolver<float> >(solver_param);

    //Load test net:
//    test_net = std::make_shared<Net<float> >(path_to_test_proto, Phase::TEST);
//    solver->net().get()->ShareTrainedLayersWith(test_net.get());

    //Load weights:
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256/best/casia_r0_iter_200000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256maxout/best/casia0_clean/_iter_125000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256maxout/best/ms_10575/ms10575_r0_iter_200000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256maxout/best/ms_12575_160_r0/ms_12575_160_r0__iter_260000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/tresnet256maxout/best/x2/orig_x2_iter_33000.caffemodel");
    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/tresnet256maxout/best/lfw_99p32/x2_ms_iter_6000.caffemodel");
//    solver->net()->CopyTrainedLayersFrom("/home/vitvit/snap/resnet256maxout/snap/ms_10575_144_r0__iter_180000.caffemodel");





    //Load L2-norm net:
    caffe::NetParameter dist_net_param;
    caffe::ReadNetParamsFromTextFileOrDie("/home/vitvit/snap/tnet1sh/l2_norm.prototxt", &dist_net_param);
//    caffe::ReadNetParamsFromTextFileOrDie("/home/vitvit/snap/tresnet512maxout/l2_norm.prototxt", &dist_net_param);
    dist_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, TRIPLET_PARAMS.dist_batch_size); //data
    dist_net_param.mutable_state()->set_phase(Phase::TEST);
    dist_net = std::make_shared<Net<float> >(dist_net_param);

    //Log:
    qDebug() << "NNTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "NNTrainer: GPU mode has been set." : "NNTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&TripletTrainerOnline::prefetch_data_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&TripletTrainerOnline::train_by_worker, this);
    train_worker.detach();
}

void TripletTrainerOnline::restore(const std::string &path_to_solver, const std::string &path_to_solverstate)
{
    if (running == true) {
        qDebug() << "NNTrainer: Training is already running.";
        return;
    }

    if (data_provider->total_identities()) {
        qDebug() << "NNTrainer: Training data are empty!";
    }

    //Load solver:
    Caffe::set_mode(Caffe::GPU);
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(path_to_solver, &solver_param);
    solver_param.set_test_initialization(false);
//    auto net_param = solver_param.mutable_net_param();
//    net_param->mutable_layer(0)->mutable_data_param()->set_batch_size(TRIPLET_PARAMS.anchor_num*(TRIPLET_PARAMS.pos_per_anchor+TRIPLET_PARAMS.neg_per_anchor+1));
    solver = std::make_shared<SGDSolver<float> >(solver_param);
//    solver->Restore(path_to_solverstate.c_str());

    //Load test net:
//    test_net = std::make_shared<Net<float> >(path_to_test_proto, Phase::TEST);
//    solver->net().get()->ShareTrainedLayersWith(test_net.get());
    solver->Restore(path_to_solverstate.c_str());

    //Load L2-norm net:
    caffe::NetParameter dist_net_param;
    caffe::ReadNetParamsFromTextFileOrDie("/home/vitvit/snap/tnet1sh/l2_norm.prototxt", &dist_net_param);
//    caffe::ReadNetParamsFromTextFileOrDie("/home/vitvit/snap/tresnet512maxout/l2_norm.prototxt", &dist_net_param);
    dist_net_param.mutable_layer(0)->mutable_input_param()->mutable_shape(0)->set_dim(0, TRIPLET_PARAMS.dist_batch_size); //data
    dist_net_param.mutable_state()->set_phase(Phase::TEST);
    dist_net = std::make_shared<Net<float> >(dist_net_param);

    //Log:
    qDebug() << "NNTrainer: Solver has been load from" << QString::fromStdString(path_to_solver);
    qDebug() << "NNTrainer: Solver's state has been restored from" << QString::fromStdString(path_to_solverstate);
    qDebug() << (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU ? "NNTrainer: GPU mode has been set." : "NNTrainer: CPU mode has been set.");

    //Run:
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&TripletTrainerOnline::prefetch_data_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&TripletTrainerOnline::train_by_worker, this);
    train_worker.detach();
}

void TripletTrainerOnline::stop()
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
}

void TripletTrainerOnline::pause()
{
//    caffe::NetParameter net_param;
//    caffe::ReadNetParamsFromTextFileOrDie("/home/vitvit/snap/_test/triplet_dropout.prototxt", &net_param);
//    net_param.mutable_state()->set_phase(Phase::TRAIN);
//    auto tnet = std::make_shared<Net<float> >(net_param);

//    std::vector<float> vec;
//    vec.resize(12*5*5, 1.0);

//    boost::shared_ptr<Blob<float> > data_blob = tnet->blob_by_name("data");
//    float* data = data_blob->mutable_cpu_data();
//    memcpy(data, vec.data(), sizeof(float)*12*5*5);

//    tnet->layer_by_name("TripletDropout")->set_phase(Phase::TEST);
//    Caffe::set_mode(Caffe::GPU);
//    tnet->Forward();

//    boost::shared_ptr<Blob<float> > output_blob = tnet->blob_by_name("output");
//    const float* output = output_blob->cpu_data();
//    for (int i = 0; i < 12; ++i) {
//        for (int j = 0; j < 5*5; ++j) {
//            qDebug() << "i" << i << ": " << output[i*(5*5)+j];
//        }
//        qDebug() << "========";
//    }

//    return;



    running = false;

    //Wait for all threads to terminate:
    while (prefetch_worker_is_running == true || train_worker_is_running == true) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    qDebug() << "NNTrainer: Training is paused. Current iter:" << solver->iter();
}

void TripletTrainerOnline::resume()
{
    running = true;

    //Preload data:
    prefetch_worker = std::thread(&TripletTrainerOnline::prefetch_data_batches_by_worker, this, prefetch_batch_margin);
    prefetch_worker.detach();

    //Run training:
    train_worker = std::thread(&TripletTrainerOnline::train_by_worker, this);
    train_worker.detach();

    qDebug() << "NNTrainer: Training is resumed.";
}
