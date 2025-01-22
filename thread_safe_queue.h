#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H
#include <queue>
#include <mutex>
#include <condition_variable>


template<typename T> class thread_safe_queue {
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable condition;

public:
    inline void emplace(const T && item) {
        {
            std::unique_lock<std::mutex> lock (this->mutex);
            this->queue.emplace(item);
        }

        this->condition.notify_one();
    }


    inline bool try_pop(T & item) {
        std::unique_lock<std::mutex> lock (this->mutex);
        if (this->queue.empty()) {
            return false;
        }
        item = this->queue.front();
        this->queue.pop();
        return true;
    }


    inline T pop() {
        std::unique_lock<std::mutex> lock (this->mutex);
        this->condition.wait(lock, [this]{return !this->queue.empty();});
        T item = this->queue.front();
        queue.pop();
        return item;
    }


    inline bool empty() {
        std::unique_lock<std::mutex> lock (this->mutex);
        return queue.empty();
    }
};



#endif //THREAD_SAFE_QUEUE_H
