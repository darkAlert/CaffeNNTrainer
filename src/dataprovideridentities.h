#ifndef DATAPROVIDERIDENTITIES_H
#define DATAPROVIDERIDENTITIES_H

#include <string>
#include <vector>
#include <deque>
#include <memory>
#include <opencv2/core.hpp>
#include <atomic>
#include <thread>
#include <mutex>
#include <boost/random.hpp>


class DataProviderIdentities
{
private:
    //Params:
    unsigned int identity_batch_size;
    unsigned int identity_batch_prefetch_size;
    unsigned int prefetch_counter;

    //Vars:
    std::vector<std::pair<std::string, int> > identity_entries_list;
    unsigned int current_identity_entries_index;
    std::deque<std::pair<std::vector<cv::Mat>, int> > identity_batch;
    std::deque<std::pair<std::vector<cv::Mat>, int> > identity_batch_prefetch;
    unsigned int current_prefetch_index;

    //Random generator:
    boost::random::mt19937 rng;                            // produces randomness out of thin air
    boost::random::uniform_int_distribution<uint64_t> die; // distribution

    //Trheads:
    std::vector<std::thread> workers;
    std::vector<bool> worker_is_running;
    std::atomic<bool> running;
    std::mutex identity_entry_mutex;
    std::mutex prefetch_batch_mutex;

    //Methods:
    unsigned char read_identities_csv(const std::string &filename);
    void shuffle_list(uint32_t times = 0);
    unsigned int shuffle_batch();
    std::string& get_identity_entry(int *label);   //identity name and label
    void read_identity_by_worker(int worker_id);
    bool prefetch(unsigned int count = 0);
    void clear();

public:
    //Methods:
    DataProviderIdentities(unsigned int identities_in_memory, unsigned int prefetch_identities_in_memory, int worker_count = 1);
    ~DataProviderIdentities();
    bool open(std::string path_to_csv);
    void stop();
    unsigned int update();  //drop old identities and add new ones

    //Getters:
    std::vector<cv::Mat>& identity_samples(unsigned int identity_index, int *label);
    std::vector<cv::Mat>& identity_samples(unsigned int identity_index, float *label);
    inline unsigned int total_identities() {
        return identity_entries_list.size();
    }
};

#endif // DATAPROVIDERIDENTITIES_H
