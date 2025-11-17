#include "SpikeEventWrapper.h"

namespace SST {
namespace SnnDL {

// 需要为虚函数提供实现以建立正确的vtable

SpikeEventWrapper::SpikeEventWrapper() : SST::Event(), spike_data(nullptr) 
{
    // 默认构造函数实现
}

SpikeEventWrapper::SpikeEventWrapper(SpikeEvent* spike) : SST::Event(), spike_data(spike) 
{
    // 参数构造函数实现
}

SpikeEventWrapper::~SpikeEventWrapper() 
{
    // 析构函数 - 不删除spike_data，由调用者管理
}

SST::Event* SpikeEventWrapper::clone() 
{
    if (spike_data) {
        // 创建SpikeEvent的副本
        SpikeEvent* cloned_spike = new SpikeEvent(
            spike_data->getNeuronId(),
            spike_data->getDestinationNeuron(),
            spike_data->getDestinationNode(),
            spike_data->getWeight(),
            spike_data->getTimestamp()
        );
        return new SpikeEventWrapper(cloned_spike);
    }
    return new SpikeEventWrapper();
}

size_t SpikeEventWrapper::size() const 
{
    return sizeof(SpikeEventWrapper) + (spike_data ? sizeof(SpikeEvent) : 0);
}

void SpikeEventWrapper::serialize_order(SST::Core::Serialization::serializer& ser) 
{
    SST::Event::serialize_order(ser);
    
    // 序列化spike_data是否存在
    bool has_spike = (spike_data != nullptr);
    ser & has_spike;
    
    if (has_spike) {
        if (ser.mode() == SST::Core::Serialization::serializer::PACK) {
            // 打包模式：序列化SpikeEvent的数据
            uint32_t neuron_id = spike_data->getNeuronId();
            uint32_t dest_neuron = spike_data->getDestinationNeuron();
            uint32_t dest_node = spike_data->getDestinationNode();
            double weight = spike_data->getWeight();
            SST::SimTime_t timestamp = spike_data->getTimestamp();
            
            ser & neuron_id;
            ser & dest_neuron;
            ser & dest_node;
            ser & weight;
            ser & timestamp;
        } else {
            // 解包模式：反序列化并创建SpikeEvent
            uint32_t neuron_id, dest_neuron, dest_node;
            double weight;
            SST::SimTime_t timestamp;
            
            ser & neuron_id;
            ser & dest_neuron;
            ser & dest_node;
            ser & weight;
            ser & timestamp;
            
            spike_data = new SpikeEvent(neuron_id, dest_neuron, dest_node, weight, timestamp);
        }
    } else if (ser.mode() == SST::Core::Serialization::serializer::UNPACK) {
        spike_data = nullptr;
    }
}

} // namespace SnnDL
} // namespace SST