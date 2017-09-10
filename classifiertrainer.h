#ifndef CLASSIFIERTRAINER_H
#define CLASSIFIERTRAINER_H

#include "dataprovidersamples.h"

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include <opencv2/core.hpp>
#include <boost/random.hpp>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>


class ClassifierTrainer
{
private:
    //Params:
    const unsigned int train_batch_size = 0;//128*1000;     //if zero then all samples will be stored in the memory
    const int test_samples_per_identity = 0;
    const int data_provider_worker_count = 2;
    const unsigned int prefetch_batch_margin = 5;
    const int test_net_batch_size = 1;

    //Vars:
    std::shared_ptr<DataProviderSamples> data_provider;
    std::shared_ptr<caffe::SGDSolver<float> > solver;
    std::shared_ptr<caffe::Net<float> > test_net;
    std::deque<std::pair<std::shared_ptr<float>, std::shared_ptr<float> > > prefetched_data;
    std::pair<std::shared_ptr<float>, std::shared_ptr<float> > test_data;   //first: images, second: labels
    unsigned int num_test_data;
    std::atomic<bool> running;
    boost::random::mt19937 rng;                            // produces randomness out of thin air

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
    ClassifierTrainer();
    ~ClassifierTrainer();

    void train(const std::string &path_to_solver, const std::string &path_to_testnet = "");
    void restore(const std::string &path_to_solver, const std::string &path_to_solverstate, const std::string &path_to_testnet = "");
    void stop();
    void pause();
    void resume();
    void openTrainData(const std::string &path_to_train);
};

#endif // CLASSIFIERTRAINER_H
