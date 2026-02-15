#pragma once

#include <string>

struct Dog {
    std::string name;

    Dog() = default;
    explicit Dog(const std::string &name) : name(name) {}

    std::string bark() const { return name + ": woof!"; }
};
