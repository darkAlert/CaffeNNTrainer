#include "dataprovidersamples.h"

#include <assert.h>
#include <fstream>
#include <opencv2/imgcodecs.hpp>           // imread

#include <QDirIterator>
#include <QDebug>


DataProviderSamples::DataProviderSamples(unsigned int samples_in_memory, int worker_count)
{
    train_batch_size = samples_in_memory;

    clear();

    running = false;
    workers.resize(worker_count > 0 ? worker_count : 1);
    worker_is_running.resize(worker_count > 0 ? worker_count : 1, false);
}

DataProviderSamples::~DataProviderSamples()
{
    stop();
    clear();
    worker_is_running.clear();
    workers.clear();
}

bool DataProviderSamples::make_sample_list(const std::string &filename, int test_samples_per_identity)
{
    clear();

    //Open CSV containing the identities list:
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        qDebug() << "DataProviderSamples: No valid input file was given, please check the given filename.";
        return false;
    }

    //Read identities csv by lines:
    std::string line;
    int label = 0;
    unsigned int id_skipped = 0, id_accepted = 0, train_samples_accepted = 0, test_samples_accepted = 0;

    while (std::getline(file, line))
    {
        int test_samples_count = 0;
        bool skipped = false;
        std::vector<std::string> path_to_identity;
        auto n = line.find(";");
        if (n != std::string::npos) {
            path_to_identity.push_back(line.substr(0,n));
            path_to_identity.push_back(line.substr(n+1,line.size()-n+1));
        }
        else {
            path_to_identity.push_back(line);
        }


        for (unsigned int i = 0; i < path_to_identity.size(); ++i)
        {
            //Read images that are contained a identity (more than 1):
            if (path_to_identity.size() > 1 || QDir(QString::fromStdString(path_to_identity[i])).entryInfoList(QStringList() << "*.PNG" << "*.png", QDir::Files).size() > 1)
            {
                QDirIterator it_files(QString::fromStdString(path_to_identity[i]), QStringList() << "*.PNG" << "*.png", QDir::Files);

                while(it_files.hasNext())
                {
                    it_files.next();
                    if (test_samples_count < test_samples_per_identity) {
                        test_entries_list.push_back(std::make_pair(it_files.filePath().toStdString(),label));
                        ++test_samples_count;
                        ++test_samples_accepted;
                    }
                    else {
                        train_entries_list.push_back(std::make_pair(it_files.filePath().toStdString(),label));
                        ++train_samples_accepted;
                    }
                }
            }
            else {
                skipped = true;
                ++id_skipped;
            }
        }
        if (skipped == false) {
            ++label;
            ++id_accepted;
        }
    }

    qDebug() << "DataProviderSamples: Total identies accepted =" << id_accepted;
    qDebug() << "DataProviderSamples: Total identities skipped =" << id_skipped;
    qDebug() << "DataProviderSamples: Total train samples accepted =" << train_samples_accepted;
    qDebug() << "DataProviderSamples: Total test samples accepted =" << test_samples_accepted;

    return true;
}

void DataProviderSamples::clear()
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

void DataProviderSamples::shuffle_entries(uint32_t times)
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

void DataProviderSamples::shuffle_batch()
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

bool DataProviderSamples::open(std::string path_to_csv, int test_samples_per_identity)
{
    if (running == true) return false;

    //Open csv:
    if (make_sample_list(path_to_csv, test_samples_per_identity) == false) {
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

bool DataProviderSamples::prefetch(int prefetch_batch_index)
{
    if (running) return false;

    prefetch_counter = train_batch_size > 0 ? train_batch_size : train_entries_list.size();
    running = true;

    //Clear batch:
    train_batch[prefetch_batch_index].clear();

    //Run workers:
    for (unsigned int i = 0; i < workers.size(); ++i) {
        worker_is_running[i] = true;
        workers[i] = std::thread(&DataProviderSamples::read_samples_by_worker, this, i, prefetch_batch_index);
        workers[i].detach();
    }

    return true;
}

void DataProviderSamples::update()
{
    if (train_batch_size == 0 && train_batch[0].empty() == false) {
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
    qDebug() << "DataProviderSamples: current batch:" << current_batch_index;

    //Prefetch new identities:
    prefetch(current_batch_index == 0 ? 1 : 0);   //this is right!!!
}

void DataProviderSamples::stop()
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

void DataProviderSamples::read_samples_by_worker(int worker_id, int prefetch_batch_index)
{
    worker_is_running[worker_id] = true;

    while (running)
    {
        //Get sample entry:
        int label;
        auto path_to_sample = get_sample_entry(&label);
        if (path_to_sample.size() == 0) continue;

        //Read a sample:
        cv::Mat sample = cv::imread(path_to_sample, -1);  //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE

        //Put images to the identity batch:
        prefetch_batch_mutex.lock();
        train_batch[prefetch_batch_index].push_back(std::make_pair(sample,label));
        prefetch_batch_mutex.unlock();
    }

    worker_is_running[worker_id] = false;
}

void DataProviderSamples::load_test_batch()
{
    test_batch.clear();

    for (unsigned int i = 0; i < test_entries_list.size(); ++i) {
        //Read a sample:
        cv::Mat sample = cv::imread(test_entries_list[i].first, -1);  //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE
        test_batch.push_back(std::make_pair(sample,test_entries_list[i].second));
    }
}

std::string& DataProviderSamples::get_sample_entry(int *label)
{
    //Lock:
    sample_entry_mutex.lock();

    //Counter of remained entries:
    if (prefetch_counter == 0) {
        running = false;
        sample_entry_mutex.unlock();
        static std::string void_str = "";
        *label = -1;
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

    *label = train_entries_list[index].second;
    return train_entries_list[index].first;    //sample path and label
}

cv::Mat& DataProviderSamples::get_train_sample(float *label)
{
    if (current_train_sample_index >= train_batch[current_batch_index].size()) {
        update();
        current_train_sample_index = 0;
    }
    auto sample_index = current_train_sample_index++;

    *label = static_cast<float>(train_batch[current_batch_index][sample_index].second);
    return train_batch[current_batch_index][sample_index].first;
}

cv::Mat& DataProviderSamples::get_test_sample(unsigned int index, float *label)
{
    if (index >= test_batch.size()) {
        assert(test_batch.size() > 0);
        index = test_batch.size()-1;
    }

    *label = static_cast<float>(test_batch[index].second);
    return test_batch[index].first;
}

void DataProviderSamples::clear_test_batch()
{
    test_batch.clear();
}
