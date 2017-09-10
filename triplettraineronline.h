#ifndef TRIPLETTRAINERONLINE_H
#define TRIPLETTRAINERONLINE_H

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"
#include <opencv2/core.hpp>
#include <boost/random.hpp>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>

#include <QString>

#include "dataprovideridentities.h"



class TripletTrainerOnline
{
private:
    //Consts:
    const unsigned int prefetch_identities_in_memory = TRIPLET_PARAMS.triplets_num*20;
    const unsigned int identities_in_memory = prefetch_identities_in_memory*20;
    const int data_provider_worker_count = 6;
    const unsigned int prefetch_batch_margin = 5;

    //Vars:
    std::shared_ptr<DataProviderIdentities> data_provider;
    std::shared_ptr<caffe::SGDSolver<float> > solver;
    std::shared_ptr<caffe::Net<float> > test_net;
    std::shared_ptr<caffe::Net<float> > dist_net;
    unsigned int current_train_data_i;

    struct TripletParams {
        //Triplet parameters:
        static constexpr float margin = 0.1;
        static const int triplets_num = 50;//32;//50;//65;
        static const int represent_mult = 10;                                          //for calculating the size of represent batch (= triplet_batch_size * represent_mult)
        static const int triplet_batch_size = triplets_num*(1+1+1);                   //for training triplet net (1+1+1 = anchor + pos + neg)
        static const int negs_num = triplet_batch_size * represent_mult - triplets_num*2;   // mult by 2 due to (anchor + positive)
        static const int dist_batch_size = triplets_num * negs_num + triplets_num;          //for computing distances
    } TRIPLET_PARAMS;

    std::deque<std::pair<std::shared_ptr<float>, std::shared_ptr<float> > > prefetched_data;
    std::deque<std::pair<std::shared_ptr<float>, std::shared_ptr<float> > > prefetched_represent_data;
    std::atomic<bool> running;
    boost::random::mt19937 rng;                            // produces randomness out of thin air
    boost::random::uniform_int_distribution<uint64_t> die; // distribution

    //Threads:
    std::mutex mutex_prefetch_data;
    std::thread prefetch_worker;
    std::thread train_worker;
    void prefetch_data_batches_by_worker(unsigned int batch_margin = 1);
    void train_by_worker();
    std::atomic<bool> prefetch_worker_is_running;
    std::atomic<bool> train_worker_is_running;

    //Methods:
    std::shared_ptr<float> set_represent_data_batch();

public:
    TripletTrainerOnline();
    ~TripletTrainerOnline();
    void train(const std::string &path_to_solver);
    void restore(const std::string &path_to_solver, const std::string &path_to_solverstate);
    void stop();
    void pause();
    void resume();
    void openTrainData(const std::string &path_to_train);
};

#endif // TRIPLETTRAINERONLINE_H
