#ifndef DATAPROVIDERIDX_H
#define DATAPROVIDERIDX_H

#include <vector>
#include <stdint.h>
#include <string>


class DataProviderIDX
{
private:
    //Vars:
    std::vector<unsigned char> raw_data;
    std::vector<float> labels;
    uint32_t sample_width, sample_height, sample_channels, sample_size, label_size;

    //Methods:
    void read_idx(std::string path_to_idx);

public:
    DataProviderIDX();
    DataProviderIDX(std::string path_to_idx);
    ~DataProviderIDX();
    void clear();
    void open(std::string path_to_idx);

    //Getters:
    inline size_t size() { return labels.size()/label_size; }
    inline uint32_t sampleWidth() { return sample_width; }
    inline uint32_t sampleHeight() { return sample_height; }
    inline uint32_t sampleChannels() { return sample_channels; }
    inline uint32_t sampleSize() { return sample_size; }
    inline uint32_t labelSize() { return label_size; }
    inline const unsigned char* sample(uint64_t index) {
        uint64_t raw_index = index*sample_size;
        if (raw_index >= raw_data.size()) return nullptr;
        return &raw_data[raw_index];
    }
    inline const float* label(uint64_t index) {
        uint64_t raw_index = index*label_size;
        if (raw_index >= labels.size()) return nullptr;
        return &labels[raw_index];
    }
};

#endif // DATAPROVIDERIDX_H
