module;

#include "BaseErrorListener.h"

#include <string>

export module Atemc.Parser.ANTLR4Parser.ErrorListener;

import Atemc.Exceptions;

export namespace atemc::parser::antlr4
{
    class AtemParserErrorListener final : public ::antlr4::BaseErrorListener
    {
    public:
        AtemParserErrorListener() = default;

        auto syntaxError(
            ::antlr4::Recognizer *recognizer,
            ::antlr4::Token *offendingSymbol,
            size_t line,
            size_t charPositionInLine,
            const std::string &msg,
            std::exception_ptr e
        ) -> void override
        {
            std::string what = msg;
            what.append("@").append(std::to_string(line)).append(":").append(std::to_string(charPositionInLine));
            throw exceptions::LexException(
                what
            );
        }
    };
}