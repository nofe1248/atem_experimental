module;

#include <stdexcept>
#include <string>

export module Atemc.Exceptions;

export namespace atemc::exceptions
{
    class AtemcAbstractException : public std::exception
    {
        const std::string msg_;
    public:
        AtemcAbstractException(std::string_view msg) : msg_(msg) {}

        auto what() const noexcept -> char const* override
        {
            return this->msg_.c_str();
        }
    };

    class ParseException : public AtemcAbstractException
    {
    public:
        using AtemcAbstractException::AtemcAbstractException;
    };

    class LexException : public AtemcAbstractException
    {
    public:
        using AtemcAbstractException::AtemcAbstractException;
    };
}
