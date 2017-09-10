#include "dataprovideridx.h"
#include <fstream>
#include <assert.h>
#include <QDebug>

const std::string idx_data_filename = "data.idx";
const std::string idx_labels_filename = "labels.idx";


DataProviderIDX::DataProviderIDX()
{
    clear();
}

DataProviderIDX::DataProviderIDX(std::string path_to_idx)
{
    clear();
    read_idx(path_to_idx);
}

DataProviderIDX::~DataProviderIDX()
{
    clear();
}


void DataProviderIDX::read_idx(std::string path_to_idx)
{
    /// Open a IDX DATA ///
    std::ifstream input_data(path_to_idx + idx_data_filename, std::ios::binary);
    input_data >> std::noskipws;                                     //need or not to need?
    std::istream_iterator<unsigned char> input_data_iterator(input_data);

    //Read width and height values (8 first bytes):
    std::vector<unsigned char> buffer;
    std::copy_n(input_data_iterator, 8, std::back_inserter(buffer));
    sample_width = ((uint32_t*)buffer.data())[0];
    sample_height = ((uint32_t*)buffer.data())[1];
    sample_channels = 1;//((uint32_t*)buffer.data())[2];
    sample_size = sample_width * sample_height * sample_channels;
    ++input_data_iterator;  //WTF?

    //Copy the rest of file to the raw_data vector:
    std::copy(input_data_iterator, std::istream_iterator<unsigned char>(), std::back_inserter(raw_data));
    input_data.close();


    /// Open a IDX LABELS ///
    std::ifstream input_labels(path_to_idx + idx_labels_filename, std::ios::binary);
    input_labels >> std::noskipws;                                     //need or not to need?
    std::istream_iterator<unsigned char> input_labels_iterator(input_labels);

    //Read the size and type of labels (8 first bytes):
    buffer.clear();
    std::copy_n(input_labels_iterator, 8, std::back_inserter(buffer));
    label_size = ((uint32_t*)buffer.data())[0];
    uint32_t label_type = ((uint32_t*)buffer.data())[1];
    ++input_labels_iterator;  //WTF?

    //Copy the rest of file to the labels vector:
    buffer.clear();
    std::copy(input_labels_iterator, std::istream_iterator<unsigned char>(), std::back_inserter(buffer));
    input_labels.close();
    labels.resize(buffer.size()/sizeof(float));
    memcpy(labels.data(), buffer.data(), buffer.size());

    assert(raw_data.size()/sample_size == labels.size()/label_size);
}

void DataProviderIDX::clear()
{
    raw_data.clear();
    labels.clear();
    sample_width = sample_height = sample_channels = sample_size = 0;
}

void DataProviderIDX::open(std::string path_to_idx)
{
    clear();
    read_idx(path_to_idx);
}
