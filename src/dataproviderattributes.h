#ifndef DATAPROVIDERATTRIBUTES_H
#define DATAPROVIDERATTRIBUTES_H

#include <string>
#include <mutex>
#include <atomic>
#include <vector>
#include <deque>
#include <thread>
#include <opencv2/core.hpp>
#include <boost/random.hpp>


class DataProviderAttributes
{
private:
    //Params:
    unsigned int train_batch_size;     //if zero then all samples will be stored in the memory
    int imread_type;                   //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE, 1: RGB
    cv::Size sample_size;

    //Vars:
    std::vector<std::pair<std::string, std::vector<float> > > train_entries_list;
    std::vector<std::pair<std::string, std::vector<float> > > test_entries_list;
    unsigned int current_train_entries_index;
    std::deque<std::pair<cv::Mat, std::vector<float> > > train_batch[2];
    std::deque<std::pair<cv::Mat, std::vector<float> > > test_batch;
    int current_batch_index;
    unsigned int prefetch_counter;
    unsigned int current_train_sample_index;
    unsigned int current_test_sample_index;

    //Random generator:
    boost::random::mt19937 rng;                            // produces randomness out of thin air
    boost::random::uniform_int_distribution<uint64_t> die; // distribution

    //Trheads:
    std::vector<std::thread> workers;
    std::vector<bool> worker_is_running;
    std::atomic<bool> running;
    std::mutex sample_entry_mutex;
    std::mutex prefetch_batch_mutex;

    //Methods:
    bool make_sample_list(const std::string &filename, std::vector<std::pair<std::string, std::vector<float> > > &sample_list);
    bool make_sample_list(const std::string &filename, int test_samples_per_identity = 0);
    void shuffle_entries(uint32_t times = 0);
    void shuffle_batch();
    bool prefetch(int prefetch_batch_index = 0);
    void clear();
    std::string& get_sample_entry(std::vector<float> &label);
    void read_samples_by_worker(int worker_id, int prefetch_batch_index);
    void load_test_batch();

public:
    DataProviderAttributes(unsigned int samples_in_memory = 0, int worker_count = 1,
                           int imread_type = -1, cv::Size sample_size = cv::Size(0,0));
    ~DataProviderAttributes();
    bool open(std::string path_to_csv, int num_test_samples = 0);
    bool open_data(const std::string &path_to_train, const std::string path_to_test = "");
    void stop();
    void update();
    void clear_test_batch();

    //Getters:
    cv::Mat& get_train_sample(std::vector<float> &label);
    cv::Mat& get_test_sample(unsigned int index, std::vector<float> &label);
    inline unsigned int total_train_samples() {
        return train_entries_list.size();
    }
    inline unsigned int total_test_samples() {
        return test_batch.size();
    }
};

#endif // DATAPROVIDERATTRIBUTES_H
