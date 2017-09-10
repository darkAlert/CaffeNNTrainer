#ifndef LANDMARKTRAINER_H
#define LANDMARKTRAINER_H

#include "dataproviderattributes.h"

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include <opencv2/core.hpp>
#include <boost/random.hpp>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>


class LandmarkTrainer
{
private:
    //Params:
    const unsigned int train_batch_size = 0;     //if zero then all samples will be stored in the memory
    const int num_test_samples = 500;
    const int data_provider_worker_count = 4;
    const unsigned int prefetch_batch_margin = 25;
    const int test_net_batch_size = 2;
    const int imread_type = 1;                   //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE, 1: RGB
    bool gpu_mode = true;

    //Vars:
    std::shared_ptr<DataProviderAttributes> data_provider;
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

    inline void swap_landmarks(std::vector<float> &label, int src_first, int src_last, int dst_first, int dst_last) {
        int num1 = abs(src_last-src_first)+1;
        int num2 = abs(dst_last-dst_first)+1;
        assert(num1 == num2);
        for (int i = 0; i < num1; ++i) {
            std::swap(label[(src_first+i)*2],label[(dst_last-i)*2]);      //x
            std::swap(label[(src_first+i)*2+1],label[(dst_last-i)*2+1]);  //y
        }
    }

public:
    LandmarkTrainer();
    ~LandmarkTrainer();

    void train(const std::string &path_to_solver);
    void restore(const std::string &path_to_solver, const std::string &path_to_solverstate);
    void stop();
    void pause();
    void resume();
    void openTrainData(const std::string &path_to_train);
};

#endif // LANDMARKTRAINER_H
