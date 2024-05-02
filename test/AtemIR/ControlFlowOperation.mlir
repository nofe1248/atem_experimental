module {
    atemir.function @main() -> i32 {
        %0 = atemir.constant #atemir.int<42>: !atemir.int<s, 64>

        atemir.while {
            atemir.yield
        } do {
            %loop_cond = atemir.constant #atemir.bool<true>: !atemir.bool
            atemir.condition(%loop_cond)
        }

        %if_cond = atemir.constant #atemir.bool<true>: !atemir.bool
        atemir.if %if_cond {
            atemir.yield
        } else {
            atemir.yield
        }

        atemir.return %0: !atemir.int<s, 64>
    }
}