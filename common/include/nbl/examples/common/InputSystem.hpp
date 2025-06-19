// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_INPUT_SYSTEM_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_INPUT_SYSTEM_HPP_INCLUDED_

namespace nbl::examples
{

class InputSystem : public core::IReferenceCounted
{
	public:
		template <class ChannelType>
		struct Channels
		{
			core::mutex lock;
			std::condition_variable added;
			core::vector<core::smart_refctd_ptr<ChannelType>> channels;
			core::vector<std::chrono::microseconds> timeStamps;
			uint32_t defaultChannelIndex = 0;
		};
		// TODO: move to "nbl/ui/InputEventChannel.h" once the interface of this utility struct matures, also maybe rename to `Consumer` ?
		template <class ChannelType>
		struct ChannelReader
		{
			template<typename F>
			inline void consumeEvents(F&& processFunc, system::logger_opt_ptr logger=nullptr)
			{
				auto events = channel->getEvents();
				const auto frontBufferCapacity = channel->getFrontBufferCapacity();
				if (events.size()>consumedCounter+frontBufferCapacity)
				{
					logger.log(
						"Detected overflow, %d unconsumed events in channel of size %d!",
						system::ILogger::ELL_ERROR,events.size()-consumedCounter,frontBufferCapacity
					);
					consumedCounter = events.size()-frontBufferCapacity;
				}
				typename ChannelType::range_t rng(events.begin() + consumedCounter, events.end());
				processFunc(rng);
				consumedCounter = events.size();
			}

			core::smart_refctd_ptr<ChannelType> channel = nullptr;
			uint64_t consumedCounter = 0ull;
		};
		
		InputSystem(system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}

		void getDefaultMouse(ChannelReader<ui::IMouseEventChannel>* reader)
		{
			getDefault(m_mouse,reader);
		}
		void getDefaultKeyboard(ChannelReader<ui::IKeyboardEventChannel>* reader)
		{
			getDefault(m_keyboard,reader);
		}
		template<class ChannelType>
		void add(Channels<ChannelType>& channels, core::smart_refctd_ptr<ChannelType>&& channel)
		{
			std::unique_lock lock(channels.lock);
			channels.channels.push_back(std::move(channel));
				
			using namespace std::chrono;
			auto timeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());
			channels.timeStamps.push_back(timeStamp);

			channels.added.notify_all();
		}
		template<class ChannelType>
		void remove(Channels<ChannelType>& channels, const ChannelType* channel)
		{
			std::unique_lock lock(channels.lock);

			auto to_remove_itr = std::find_if(
					channels.channels.begin(),channels.channels.end(),[channel](const auto& chan)->bool{return chan.get()==channel;}
			);

			auto index = std::distance(channels.channels.begin(), to_remove_itr);

			channels.timeStamps.erase(channels.timeStamps.begin() + index);
			channels.channels.erase(to_remove_itr);
		}
		template<class ChannelType>
		void getDefault(Channels<ChannelType>& channels, ChannelReader<ChannelType>* reader)
		{
			/*
			* TODO: Improve default device switching.
			* For nice results, we should actually make a multi-channel reader,
			* and then keep a consumed counter together with a last consumed event from each channel.
			* If there is considerable pause in events received by our current chosen channel or
			* we can detect some other channel of the same "compatible class" is producing more events,
			* Switch the channel choice, but prune away all events younger than the old default's consumption timestamp.
			* (Basically switch keyboards but dont try to process events older than the events you've processed from the old keyboard)
			*/
				
			std::unique_lock lock(channels.lock);
			while (channels.channels.empty())
			{
				m_logger.log("Waiting For Input Device to be connected...",system::ILogger::ELL_INFO);
				channels.added.wait(lock);
			}
				
			uint64_t consumedCounter = 0ull;

			using namespace std::chrono;
			constexpr long long DefaultChannelTimeoutInMicroSeconds = 100*1e3; // 100 mili-seconds
			auto nowTimeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());

			// Update Timestamp of all channels
			for(uint32_t ch = 0u; ch < channels.channels.size(); ++ch) {
				auto & channel = channels.channels[ch];
				auto & timeStamp = channels.timeStamps[ch];
				auto events = channel->getEvents();
				if(events.size() > 0) {
					auto lastEventTimeStamp = (*(events.end() - 1)).timeStamp; // last event timestamp
					timeStamp = lastEventTimeStamp;
				}
			}

			auto defaultIdx = channels.defaultChannelIndex;
			if(defaultIdx >= channels.channels.size()) {
				defaultIdx = 0;
			}
			auto defaultChannel = channels.channels[defaultIdx];
			auto defaultChannelEvents = defaultChannel->getEvents();
			auto timeDiff = (nowTimeStamp - channels.timeStamps[defaultIdx]).count();
				
			constexpr size_t RewindBackEvents = 50u;

			// If the current one hasn't been active for a while
			if(defaultChannel->empty()) {
				if(timeDiff > DefaultChannelTimeoutInMicroSeconds) {
					// Look for the most active channel (the channel which has got the most events recently)
					auto newDefaultIdx = defaultIdx;
					microseconds maxEventTimeStamp = microseconds(0);

					for(uint32_t chIdx = 0; chIdx < channels.channels.size(); ++chIdx) {
						if(defaultIdx != chIdx) 
						{
							auto channelTimeDiff = (nowTimeStamp - channels.timeStamps[chIdx]).count();
							// Check if was more recently active than the current most active
							if(channelTimeDiff < DefaultChannelTimeoutInMicroSeconds)
							{
								auto & channel = channels.channels[chIdx];
								auto channelEvents = channel->getEvents();
								auto channelEventSize = channelEvents.size();
								const auto frontBufferCapacity = channel->getFrontBufferCapacity();

								size_t rewindBack = std::min(RewindBackEvents, frontBufferCapacity);
								rewindBack = std::min(rewindBack, channelEventSize);

								auto oldEvent = *(channelEvents.end() - rewindBack);

								// Which oldEvent of channels are most recent.
								if(oldEvent.timeStamp > maxEventTimeStamp) {
									maxEventTimeStamp = oldEvent.timeStamp;
									newDefaultIdx = chIdx;
								}
							}
						}
					}

					if(defaultIdx != newDefaultIdx) {
						m_logger.log("Default InputChannel for ChannelType changed from %u to %u",system::ILogger::ELL_INFO, defaultIdx, newDefaultIdx);

						defaultIdx = newDefaultIdx;
						channels.defaultChannelIndex = newDefaultIdx;
						defaultChannel = channels.channels[newDefaultIdx];
							
						consumedCounter = defaultChannel->getEvents().size() - defaultChannel->getFrontBufferCapacity(); // to not get overflow in reader when consuming.
					}
				}
			}

			if (reader->channel==defaultChannel)
				return;

			reader->channel = defaultChannel;
			reader->consumedCounter = consumedCounter;
		}

		system::logger_opt_smart_ptr m_logger;
		Channels<ui::IMouseEventChannel> m_mouse;
		Channels<ui::IKeyboardEventChannel> m_keyboard;
};

}
#endif
