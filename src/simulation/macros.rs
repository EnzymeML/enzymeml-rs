#[macro_export]
macro_rules! create_stoich_eq_jit {
    ($jit_fun:expr, $stoich:expr, $non_zero_indices:expr, $law_output_size:expr) => {{
        let stoich = $stoich;
        let non_zero_indices = $non_zero_indices;
        let law_output_size = $law_output_size;

        let jjit_fun: $crate::simulation::system::EvalFunction = std::sync::Arc::new(move |input: &[f64], output: &mut [f64]| {
            // Use thread-local storage for the buffer to avoid allocations during simulation
            thread_local! {
                static LAW_OUTPUT: std::cell::RefCell<Vec<f64>> = std::cell::RefCell::new(Vec::new());
            }

            LAW_OUTPUT.with(|buffer| {
                let mut law_output = buffer.borrow_mut();

                // Resize only if necessary
                if law_output.len() != law_output_size {
                    law_output.resize(law_output_size, 0.0);
                } else {
                    // Fast clear - fill with zeros
                    law_output.fill(0.0);
                }

                // Calculate reaction rates
                $jit_fun.fun()(input, &mut law_output);

                // Calculate dot manually
                for &(i, j) in &non_zero_indices {
                    output[i] += law_output[j] * stoich[(i, j)];
                }
            });
        });

        jjit_fun
    }};
}
