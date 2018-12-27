#include <iostream>
#include <string>
#include <unordered_map>
#include <list>
#include <typeinfo>

template<typename K, typename V>
class LRUCache {
private:
    using Node = std::pair<V, K>;
    using DType = std::unordered_map<K, typename std::list<Node>::iterator>;
    using LType = std::list<Node>;
    DType cachedict;
    LType cachelist;
    size_t capacity;

    std::tuple<bool, typename DType::iterator> is_present(K key){
        auto keysearch = cachedict.find(key);
        return std::make_tuple(keysearch != cachedict.end(), keysearch);
    }
public:
    LRUCache(size_t cap) : capacity(cap) {}

    std::tuple<bool, V> get(K key) { //TODO: std::optional?
        bool present; typename DType::iterator dict_iter;
        std::tie(present, dict_iter) = is_present(key);
        V val; K back_key;
        if (present){
            std::tie(val, back_key) = *(dict_iter->second);  
            // move it to the front of the list
            cachelist.erase(dict_iter->second);
            cachelist.push_front({val, back_key});
            cachedict[key] = cachelist.begin();
        }
        return {present, val};
    }
    void put(K key, V value) {
        V val; K back_key;
        if (cachelist.size() == capacity){
            std::tie(val, back_key) = cachelist.back();
            cachelist.pop_back();
            cachedict.erase(back_key);
        }
        cachelist.push_front({value, key});
        cachedict[key] = cachelist.begin();
    }

    void printme(){
        //std::cout << cache[1] << "\n";
        for (auto it : cachedict){
           std::cout << " " << it.first << ": " << it.second->first << "; ";
        }
        for (auto it : cachelist){
            std::cout << "(" << it.first << " " << it.second << "); ";
        }
        std::cout << "\n";
    }
};

int main() {
    LRUCache<int, int> d = LRUCache<int, int>(3);
    d.put(1,1);
    d.printme();
    std::cout << std::get<1>(d.get(1)) << "\n";
    d.printme();
    d.put(2,2);
    d.put(3,3);
    //d.printme();
    std::cout << std::get<1>(d.get(1)) << "\n";
    d.put(4,4);
    std::cout << std::get<1>(d.get(3)) << "\n";
    std::cout << std::get<1>(d.get(3)) << "\n";
    d.printme();
    std::cout << std::get<0>(d.get(2)) << " " << std::get<1>(d.get(2)) << "\n";
    std::cout << std::get<1>(d.get(1)) << "\n";
    std::cout << std::get<1>(d.get(4)) << "\n";
    std::cout << std::get<1>(d.get(2)) << "\n";
    std::cout << std::get<1>(d.get(3)) << "\n";


    std::cout << "Bye\n";
}