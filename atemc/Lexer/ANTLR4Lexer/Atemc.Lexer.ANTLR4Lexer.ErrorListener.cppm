module;

#include "BaseErrorListener.h"

#include <string>
#include <format>

export module Atemc.Lexer.ANTLR4Lexer.ErrorListener;

import Atemc.Exceptions;

export namespace atemc::lexer::antlr4
{
    class AtemLexerErrorListener final : public ::antlr4::BaseErrorListener
    {
    public:
        AtemLexerErrorListener() = default;

        auto syntaxError(
            ::antlr4::Recognizer *recognizer,
            ::antlr4::Token *offendingSymbol,
            size_t line,
            size_t charPositionInLine,
            std::string const &msg,
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