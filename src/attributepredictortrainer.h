#ifndef ATTRIBUTEPREDICTORTRAINER_H
#define ATTRIBUTEPREDICTORTRAINER_H

#include "dataproviderattributes.h"

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include <opencv2/core.hpp>
#include <boost/random.hpp>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>


class AttributePredictorTrainer
{
private:
    //Params:
    const unsigned int train_batch_size = 20000;//96*100;     //if zero then all samples will be stored in the memory
    const int num_test_samples = 0;//1000;
    const int data_provider_worker_count = 4;//4;
    const unsigned int prefetch_batch_margin = 100;
    const int test_net_batch_size = 1;//20;
    const int imread_type = 0;                   //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE, 1: RGB

    //Vars:
    std::shared_ptr<DataProviderAttributes> data_provider;
    std::shared_ptr<caffe::SGDSolver<float> > solver;
    std::shared_ptr<caffe::Net<float> > test_net;
    std::deque<std::pair<std::shared_ptr<float>, std::shared_ptr<float> > > prefetched_data;
    std::pair<std::shared_ptr<float>, std::shared_ptr<float> > test_data;   //first: images, second: labels
    unsigned int num_test_data;
    std::atomic<bool> running;
    boost::random::mt19937 rng;                            // produces randomness out of thin air
    bool gpu_mode;

    //Threads:
    std::mutex mutex_prefetch_data;
    std::thread prefetch_worker;
    std::thread train_worker;
    std::atomic<bool> prefetch_worker_is_running;
    std::atomic<bool> train_worker_is_running;
    void prefetch_batches_by_worker(unsigned int batch_margin = 1);
    void train_by_worker();

    //Methods:
    void set_data_batch();
    void set_test_data_batch(int sample_index);
    void prepare_test_data();

public:
    AttributePredictorTrainer(bool gpu_mode = true);
    ~AttributePredictorTrainer();

    void train(const std::string &path_to_solver);
    void restore(const std::string &path_to_solver, const std::string &path_to_solverstate);
    void stop();
    void pause();
    void resume();
    void openTrainData(const std::string &path_to_train);
};

#endif // ATTRIBUTEPREDICTORTRAINER_H
