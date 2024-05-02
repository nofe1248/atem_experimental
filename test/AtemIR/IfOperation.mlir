module {
    atemir.function @main() -> !atemir.int<s, 64> {
        %cond = atemir.constant #atemir.bool<true>: !atemir.bool
        atemir.if %cond {
            atemir.yield
        } else {
            atemir.yield
        }
        atemir.return %cond: !atemir.bool
    }
}