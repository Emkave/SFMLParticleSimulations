#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <condition_variable>

class thread_pool {
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    bool stop = false;
    std::condition_variable condition;
    std::mutex mutex;

public:
    explicit thread_pool(const unsigned short thread_num = 20);
    ~thread_pool();

    inline void enqueue(const std::function<void()> & task);
};


inline thread_pool::thread_pool(const unsigned short thread_num) {
    this->threads.reserve(thread_num);

    for (unsigned short i=0; i<thread_num; i++) {
        this->threads.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock (this->mutex);
                    this->condition.wait(lock, [this]{return this->stop || !this->tasks.empty();});

                    if (this->stop && this->tasks.empty()) {
                        return;
                    }

                    if (!this->tasks.empty()) {
                        task = static_cast<std::function<void()>&&>(this->tasks.front());
                        this->tasks.pop();
                    }
                }

                task();
            }
        });
    }
}


inline thread_pool::~thread_pool() {
    this->stop = true;
    this->condition.notify_all();

    for (unsigned short i=0; i<this->threads.size(); i++) {
        this->threads[i].join();
    }
}


inline void thread_pool::enqueue(const std::function<void()> & task) {
    {
        std::unique_lock<std::mutex> lock (this->mutex);

        if (this->stop) {
            return;
        }

        this->tasks.emplace(task);
    }

    this->condition.notify_one();
}

#endif //THREAD_POOL_H
