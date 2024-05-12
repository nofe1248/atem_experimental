module {
    atemir.function @main() -> !atemir.int<s, 64> {
        %0 = atemir.constant #atemir.int<42>: !atemir.int<s, 64>

        atemir.while {
            atemir.yield
        } do {
            %loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%loop_cond)
        }

        atemir.return %0: !atemir.int<s, 64>
    }
}