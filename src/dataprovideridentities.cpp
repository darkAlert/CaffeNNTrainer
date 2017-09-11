#include "dataprovideridentities.h"

#include <assert.h>
#include <fstream>
#include <opencv2/imgcodecs.hpp>           // imread

#include <QDebug>
#include <QFileInfo>
#include <QFile>
#include <QDir>
#include <QDirIterator>

DataProviderIdentities::DataProviderIdentities(unsigned int identities_in_memory, unsigned int prefetch_identities_in_memory, int worker_count)
{
    assert (worker_count > 0);
    identity_batch_size = identities_in_memory;
    identity_batch_prefetch_size = prefetch_identities_in_memory;
    if (prefetch_identities_in_memory > 0) {
        assert (prefetch_identities_in_memory <= identities_in_memory);
    }

    clear();
    running = false;

    workers.resize(worker_count);
    worker_is_running.resize(worker_count, false);
}

DataProviderIdentities::~DataProviderIdentities()
{
    stop();
    clear();
    worker_is_running.clear();
    workers.clear();
}

unsigned char DataProviderIdentities::read_identities_csv(const std::string &filename)
{
    //Open CSV:
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file) {
        qDebug() << "DataProviderIdentities: No valid input file was given, please check the given filename.";
        return 0;
    }

    //Allocate memory:
    clear();
    identity_entries_list.reserve(std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n'));
    file.clear();
    file.seekg(0, std::ios::beg);  //back to the beginning

    //Read by lines:
    std::string line;
    QStringList filter;
    filter << "*.png";
    unsigned int skipped = 0;
    int label = 0;

    while (std::getline(file, line)) {
        QString path = QString::fromStdString(line);
        if (QDir(path).entryInfoList(filter, QDir::Files).size() > 1) {   //only identities containing more 1 image
            identity_entries_list.push_back(std::make_pair(line,label++));
        }
        else {
            ++skipped;
        }
    }
    qDebug() << "DataProviderIdentities: Total identities skipped =" << skipped;
    qDebug() << "DataProviderIdentities: total identies =" << identity_entries_list.size();

    return identity_entries_list.size();
}

void DataProviderIdentities::clear()
{
    identity_entries_list.clear();
    current_identity_entries_index = 0;
    identity_batch.clear();
    current_prefetch_index = 0;
    prefetch_counter = 0;
}

void DataProviderIdentities::shuffle_list(uint32_t times)
{
    if (times == 0) times = identity_entries_list.size();
    rng.seed(std::time(nullptr));
    die = boost::random::uniform_int_distribution<uint64_t> (0, identity_entries_list.size() - 1);

    for (uint32_t k = 0; k < times; ++k) {
        auto i = die(rng);
        auto j = die(rng);
        if (i == j) continue;
        std::swap(identity_entries_list[i], identity_entries_list[j]);
    }
}

unsigned int DataProviderIdentities::shuffle_batch()
{
    auto times = identity_batch.size();
    rng.seed(std::time(nullptr));
    die = boost::random::uniform_int_distribution<uint64_t> (0, identity_batch.size() - 1);

    for (uint32_t k = 0; k < times; ++k) {
        auto i = die(rng);
        auto j = die(rng);
        if (i == j) continue;
        std::swap(identity_batch[i], identity_batch[j]);
    }

    return identity_batch.size();
}

std::string& DataProviderIdentities::get_identity_entry(int *label)
{
    //Lock:
    identity_entry_mutex.lock();

    //Counter of remained entries:
    if (prefetch_counter == 0) {
        running = false;
        identity_entry_mutex.unlock();
        static std::string void_str = "";
        *label = -1;
        return void_str;
    }
    --prefetch_counter;
    //Shuffle if need:
    if (current_identity_entries_index >= identity_entries_list.size()) {
        shuffle_list();
        current_identity_entries_index = 0;
    }
    //Get index of current entry:
    auto index = current_identity_entries_index;
    ++current_identity_entries_index;

    //Unlock:
    identity_entry_mutex.unlock();

    *label = identity_entries_list[index].second;
    return identity_entries_list[index].first;    //identity name and label
}

void DataProviderIdentities::read_identity_by_worker(int worker_id)
{
    worker_is_running[worker_id] = true;

    QStringList filter;
    filter << "*.png" << "*.jpg" << "*.PNG";

    while (running)
    {
        //Get identity:
        int label;
        auto entry = get_identity_entry(&label);
        if (entry.size() == 0) continue;
        QString path_to_idenity = QString("%1/").arg(QString::fromStdString(entry));

        //Read the idenity images:
        std::vector<cv::Mat> images;
        images.reserve(QDir(path_to_idenity).entryInfoList(filter, QDir::Files).size());
        QDirIterator it_files(path_to_idenity, filter, QDir::Files);

        while(it_files.hasNext()) {
            it_files.next();
            QString path_to_img = it_files.filePath();
            images.push_back(cv::imread(path_to_img.toStdString(), -1));  //-1: IMREAD_UNCHANGED, 0: CV_IMREAD_GRAYSCALE
        }

        //Put images to the identity batch:
        prefetch_batch_mutex.lock();
        identity_batch_prefetch.push_back(std::make_pair(images,label));
        prefetch_batch_mutex.unlock();
    }

    worker_is_running[worker_id] = false;
}

bool DataProviderIdentities::prefetch(unsigned int count)
{
    if (running) return false;

    prefetch_counter = count > 0 ? count : identity_batch_prefetch_size;
    running = true;

    //Run workers:
    for (unsigned int i = 0; i < workers.size(); ++i) {
        worker_is_running[i] = true;
        workers[i] = std::thread(&DataProviderIdentities::read_identity_by_worker, this, i);
        workers[i].detach();
    }

    return true;
}

unsigned int DataProviderIdentities::update()
{
    if (identity_batch_prefetch_size == 0 && identity_batch.empty() == false) {
        return shuffle_batch();
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

    //Pop old identities from identity_batch:
    unsigned int size = identity_batch.size() >= identity_batch_prefetch.size() ? identity_batch.size() - identity_batch_prefetch.size() : 0;
    while (identity_batch.size() > size) {
        identity_batch.pop_front();
    }

    //Insert a prefetch batch to the identity one:
    unsigned updated = identity_batch_prefetch.size();
    identity_batch.insert(identity_batch.end(),identity_batch_prefetch.begin(), identity_batch_prefetch.end());
    identity_batch_prefetch.clear();

    //Prefetch new identities:
    prefetch();

    return updated;
}


bool DataProviderIdentities::open(std::string path_to_csv)
{
    if (running == true) return false;

    //Open csv:
    if (read_identities_csv(path_to_csv) == 0) {
        return false;
    }
    assert(identity_batch_size <= total_identities());

    //Fill first identity batch:
    current_identity_entries_index = identity_entries_list.size(); //to shuffle in future
    if (prefetch(identity_batch_size) == false) return false;
    update();

    return true;
}

void DataProviderIdentities::stop()
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

std::vector<cv::Mat>& DataProviderIdentities::identity_samples(unsigned int identity_index, int *label)
{
    if (identity_index >= identity_batch.size()) {
        identity_index = identity_batch.size()-1;
    }

    *label = identity_batch[identity_index].second;
    return identity_batch[identity_index].first;
}

std::vector<cv::Mat>& DataProviderIdentities::identity_samples(unsigned int identity_index, float *label)
{
    if (identity_index >= identity_batch.size()) {
        assert(identity_batch.size() > 0);
        identity_index = identity_batch.size()-1;
    }

    *label = static_cast<float>(identity_batch[identity_index].second);
    return identity_batch[identity_index].first;
}
