#include "dataproviderattributes.h"

#include <assert.h>
#include <fstream>
#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist

#include <QFileInfo>
#include <QDebug>


DataProviderAttributes::DataProviderAttributes(unsigned int samples_in_memory, int worker_count,
                                               int imread_type, cv::Size sample_size)
    : imread_type (imread_type), sample_size(sample_size)
{
    train_batch_size = samples_in_memory;

    clear();

    running = false;
    workers.resize(worker_count > 0 ? worker_count : 1);
    worker_is_running.resize(worker_count > 0 ? worker_count : 1, false);
}

DataProviderAttributes::~DataProviderAttributes()
{
    stop();
    clear();
    worker_is_running.clear();
    workers.clear();
}

bool DataProviderAttributes::make_sample_list(const std::string &filename, std::vector<std::pair<std::string, std::vector<float> > > &sample_list)
{
    sample_list.clear();

    //Open CSV containing the identities list:
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        qDebug() << "DataProviderAttributes: No valid input file was given, please check the given filename.";
        return false;
    }

    //Read identities csv by lines:
    std::string line;
    int missed = 0;

    while (std::getline(file, line))
    {
        //Parse:
        std::string path_to_sample;
        std::vector<float> label;
        size_t prev_pos = 0, pos = 0;
        //Parse the path to a sample:
        pos = line.find(";", prev_pos);
        path_to_sample = line.substr(0,pos-0);
        prev_pos = pos+1;
        //Parse the attribute label:
        while ((pos = line.find(";", prev_pos)) != std::string::npos) {
            label.push_back(std::stof(line.substr(prev_pos,pos-prev_pos)));
            prev_pos = pos+1;
        }
        label.push_back(std::stof(line.substr(prev_pos)));

        //Check whether the image exist:
        if (path_to_sample.size() > 0 && QFileInfo(QString::fromStdString(path_to_sample)).exists()) {
            sample_list.push_back(std::make_pair(path_to_sample, label));
        }
        else {
            ++missed;
        }
    }

    if (missed > 0) {
        qDebug() << "DataProviderAttributes<warning>: Some samples were not found. Total missed:" << missed;
    }

    return true;
}

bool DataProviderAttributes::make_sample_list(const std::string &filename, int num_test_samples)
{
    clear();

    //Open CSV containing the identities list:
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        qDebug() << "DataProviderAttributes: No valid input file was given, please check the given filename.";
        return false;
    }

    //Read identities csv by lines:
    std::string line;
    int samples_skipped = 0, train_samples_accepted = 0, test_samples_accepted = 0;

    while (std::getline(file, line))
    {
        //Parse:
        std::string path_to_sample;
        std::vector<float> label;
        size_t prev_pos = 0, pos = 0;
        //Parse the path to a sample:
        pos = line.find(";", prev_pos);
        path_to_sample = line.substr(0,pos-0);
        prev_pos = pos+1;
        //Parse the attribute label:
        while ((pos = line.find(";", prev_pos)) != std::string::npos) {
            label.push_back(std::stof(line.substr(prev_pos,pos-prev_pos)));
            prev_pos = pos+1;
        }
        label.push_back(std::stof(line.substr(prev_pos)));

        //Check whether the image exist:
        if (path_to_sample.size() > 0 && QFileInfo(QString::fromStdString(path_to_sample)).exists()) {
            //For test:
            if (test_samples_accepted < num_test_samples) {
                test_entries_list.push_back(std::make_pair(path_to_sample, label));
                ++test_samples_accepted;
            }
            //For train:
            else {
                train_entries_list.push_back(std::make_pair(path_to_sample, label));
                ++train_samples_accepted;
            }
        }
        else {
            ++samples_skipped;
        }
    }

    qDebug() << "DataProviderAttributes: Total samples skipped =" << samples_skipped;
    qDebug() << "DataProviderAttributes: Total train samples accepted =" << train_samples_accepted;
    qDebug() << "DataProviderAttributes: Total test samples accepted =" << test_samples_accepted;

    return true;
}

void DataProviderAttributes::clear()
{
    train_batch[0].clear();
    train_batch[1].clear();
    test_batch.clear();
    train_entries_list.clear();
    test_entries_list.clear();
    current_train_entries_index = 0;
    prefetch_counter = 0;
    current_train_sample_index = 0;
    current_test_sample_index = 0;
    current_batch_index = 0;
}

void DataProviderAttributes::shuffle_entries(uint32_t times)
{
    if (times == 0) times = train_entries_list.size();
    rng.seed(std::time(nullptr));
    die = boost::random::uniform_int_distribution<uint64_t> (0, train_entries_list.size() - 1);

    for (uint32_t k = 0; k < times; ++k) {
        auto i = die(rng);
        auto j = die(rng);
        if (i == j) continue;
        std::swap(train_entries_list[i], train_entries_list[j]);
    }
}

void DataProviderAttributes::shuffle_batch()
{
    auto times = train_batch[0].size();
    rng.seed(std::time(nullptr));
    die = boost::random::uniform_int_distribution<uint64_t> (0, train_batch[0].size() - 1);

    for (uint32_t k = 0; k < times; ++k) {
        auto i = die(rng);
        auto j = die(rng);
        if (i == j) continue;
        std::swap(train_batch[0][i], train_batch[0][j]);
    }
}

bool DataProviderAttributes::open_data(const std::string &path_to_train, const std::string path_to_test)
{
    if (running == true) return false;
    clear();

    //Open train data:
    if (make_sample_list(path_to_train, train_entries_list) == false) {
        return false;
    }
    qDebug() << "DataProviderAttributes: Training samples have been loaded. Total count:" << train_entries_list.size();
    assert(train_batch_size <= train_entries_list.size());

    //Open test data (if needed):
    if (path_to_test.length() > 0) {
        if (make_sample_list(path_to_test, test_entries_list) == false) {
            return false;
        }
    }
    qDebug() << "DataProviderAttributes: Testing samples have been loaded. Total count:" << test_entries_list.size();

    //Load test batch:
    load_test_batch();

    //Fill first identity batch:
    current_train_entries_index = train_entries_list.size(); //to shuffle in future
    if (prefetch(0) == false) return false;
    current_batch_index = train_batch_size > 0 ? 1 : 0;
    update();

    return true;
}

//LEGACY data openning:
bool DataProviderAttributes::open(std::string path_to_csv, int num_test_samples)
{
    if (running == true) return false;

    //Open csv:
    if (make_sample_list(path_to_csv, num_test_samples) == false) {
        return false;
    }
    assert(train_batch_size <= train_entries_list.size());

    //Load test batch:
    load_test_batch();

    //Fill first identity batch:
    current_train_entries_index = train_entries_list.size(); //to shuffle in future
    if (prefetch(0) == false) return false;
    current_batch_index = train_batch_size > 0 ? 1 : 0;
    update();

    return true;
}

bool DataProviderAttributes::prefetch(int prefetch_batch_index)
{
    if (running) return false;

    prefetch_counter = train_batch_size > 0 ? train_batch_size : train_entries_list.size();
    running = true;

    //Clear batch:
    train_batch[prefetch_batch_index].clear();

    //Run workers:
    for (unsigned int i = 0; i < workers.size(); ++i) {
        worker_is_running[i] = true;
        workers[i] = std::thread(&DataProviderAttributes::read_samples_by_worker, this, i, prefetch_batch_index);
        workers[i].detach();
    }

    return true;
}

void DataProviderAttributes::update()
{
    if (train_batch_size == 0 && train_batch[0].empty() == false) {
        current_batch_index = 0;
        shuffle_batch();
        return;
    }

    //Wait for all threads to terminate:
    bool all_were_terminated;
    do {
        all_were_terminated = true;
        for (unsigned int i = 0; i < workers.size(); ++i) {
            if (worker_is_running[i] == true) {
                all_were_terminated = false;
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
                break;
            }
        }
    } while (all_were_terminated == false);

    //Change current batch:
    current_batch_index = current_batch_index == 0 ? 1 : 0;
//    qDebug() << "DataProviderAttributes: current batch:" << current_batch_index;

    //Prefetch new identities:
    prefetch(current_batch_index == 0 ? 1 : 0);   //this is right!!!
}

void DataProviderAttributes::stop()
{
    running = false;

    //Wait for all threads to terminate:
    bool all_were_terminated;
    do {
        all_were_terminated = true;
        for (unsigned int i = 0; i < workers.size(); ++i) {
            if (worker_is_running[i] == true) {
                all_were_terminated = false;
                std::this_thread::sleep_for(std::chrono::microseconds(1000));
                break;
            }
        }
    } while (all_were_terminated == false);

    clear();
}

void DataProviderAttributes::read_samples_by_worker(int worker_id, int prefetch_batch_index)
{
    worker_is_running[worker_id] = true;

    while (running)
    {
        //Get sample entry:
        std::vector<float> label;
        auto path_to_sample = get_sample_entry(label);
        if (path_to_sample.size() == 0) continue;

        //Read a sample:
        cv::Mat sample = cv::imread(path_to_sample, imread_type);  //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE
        if (sample_size.width > 0 && sample_size.height > 0) {
            auto inter = sample.cols > sample_size.width ? CV_INTER_AREA : CV_INTER_LINEAR;
            cv::resize(sample, sample, sample_size, 0, 0, inter);
        }

//        //preprocessing:
//        auto inter = sample.cols > 150 ? CV_INTER_AREA : CV_INTER_LINEAR;
//        cv::Mat temp;
//        cv::resize(sample,temp,cv::Size(150, 200), 0, 0, inter);
//        sample = temp(cv::Rect(0,25,150,150)).clone();
//        cv::Mat temp;
//        cv::threshold(sample, sample, 1, 255, CV_THRESH_BINARY);
//        cv::merge(std::vector<cv::Mat>({temp,temp,temp}), sample);
//        qDebug() << "after:" << sample.channels();

        //Put images to the identity batch:
        prefetch_batch_mutex.lock();
        train_batch[prefetch_batch_index].push_back(std::make_pair(sample,label));
        prefetch_batch_mutex.unlock();
    }

    worker_is_running[worker_id] = false;
}

void DataProviderAttributes::load_test_batch()
{
    test_batch.clear();

    for (unsigned int i = 0; i < test_entries_list.size(); ++i) {
        //Read a sample:
        cv::Mat sample = cv::imread(test_entries_list[i].first, imread_type);  //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE
        if (sample_size.width > 0 && sample_size.height > 0) {
            auto inter = sample.cols > sample_size.width ? CV_INTER_AREA : CV_INTER_LINEAR;
            cv::resize(sample, sample, sample_size, 0, 0, inter);
        }

//        //preprocessing:
//        auto inter = sample.cols > 150 ? CV_INTER_AREA : CV_INTER_LINEAR;
//        cv::Mat temp;
//        cv::resize(sample,temp,cv::Size(150, 200), 0, 0, inter);
//        sample = temp(cv::Rect(0,25,150,150)).clone();

        test_batch.push_back(std::make_pair(sample,test_entries_list[i].second));
    }
}

std::string& DataProviderAttributes::get_sample_entry(std::vector<float> &label)
{
    //Lock:
    sample_entry_mutex.lock();

    //Counter of remained entries:
    if (prefetch_counter == 0) {
        running = false;
        sample_entry_mutex.unlock();
        static std::string void_str = "";
        return void_str;
    }
    --prefetch_counter;
    //Shuffle if need:
    if (current_train_entries_index >= train_entries_list.size()) {
        shuffle_entries();
        current_train_entries_index = 0;
    }
    //Get index of current sample:
    auto index = current_train_entries_index;
    ++current_train_entries_index;

    //Unlock:
    sample_entry_mutex.unlock();

    label = train_entries_list[index].second;
    return train_entries_list[index].first;    //sample path and label
}

cv::Mat& DataProviderAttributes::get_train_sample(std::vector<float> &label)
{
    if (current_train_sample_index >= train_batch[current_batch_index].size()) {
        update();
        current_train_sample_index = 0;
    }
    auto sample_index = current_train_sample_index++;

    label = train_batch[current_batch_index][sample_index].second;
    return train_batch[current_batch_index][sample_index].first;
}

cv::Mat& DataProviderAttributes::get_test_sample(unsigned int index, std::vector<float> &label)
{
    if (index >= test_batch.size()) {
        assert(test_batch.size() > 0);
        index = test_batch.size()-1;
    }

    label = test_batch[index].second;
    return test_batch[index].first;
}

void DataProviderAttributes::clear_test_batch()
{
    test_batch.clear();
}
