#ifndef FCTRAINER_H
#define FCTRAINER_H

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include <opencv2/core.hpp>
#include <boost/random.hpp>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>


class FCTrainer
{
private:
    //Params:
    const bool gpu_mode = true;
    const unsigned int prefetch_batch_margin = 25;
    const unsigned int num_test_samples = 5;

    //Vars:
    std::shared_ptr<caffe::SGDSolver<float> > solver;
    std::shared_ptr<caffe::Net<float> > test_net;
    std::deque<std::pair<std::shared_ptr<float>, std::shared_ptr<float> > > prefetched_data;
    std::pair<std::shared_ptr<float>, std::shared_ptr<float> > test_data;   //first: images, second: labels
    std::vector<std::pair<std::vector<float>, std::vector<float> > > raw_train_samples;
    std::vector<std::pair<std::vector<float>, std::vector<float> > > raw_test_samples;
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
    FCTrainer();
    ~FCTrainer();

    void train(const std::string &path_to_solver);
    void restore(const std::string &path_to_solver, const std::string &path_to_solverstate);
    void stop();
    void pause();
    void resume();
    bool read_raw_samples(const std::string &path_to_csv);
};

#endif // FCTRAINER_H
