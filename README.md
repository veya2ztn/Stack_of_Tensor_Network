###**Batch TensorNetwork**

We consider the scenarios that 

- There is a core tensor network as model's weight $W$
- There are hundreds of tensor network $B_i$ as input data

Our goal is to calculate the hundreds of inner product  $\lang W|B_i\rang$ as fast as possible.

<img src="https://github.com/veya2ztn/TNproject/blob/main/benchmark/BatchContraction/figures/diagram1.png" alt="diagram1" style="zoom:50%;" />

There are three way to make it:

- a naïve `for` loop and try to call parallel processing to speed up
- use a `batch` version of the TensorNetwork.
- use `einsum` to achieve `efficient coding`.

------

For the `batch` version of TensorNetwork, please see our paper(coming soon). The conclusion is very simple, the `batch` is another `blockdiag` TensorNetwork with larger bond dimension.

![image-20220124112616361](https://github.com/veya2ztn/TNproject/blob/main/benchmark/BatchContraction/figures/benchmark.png)

The `einsum` can hold `batch contraction` like `Babcd,Babcd->B` which is not a tensor contraction (so it cannot be replicated by `tensordot`). It is same as treat the `batch ` TensorNetwork as sparse matrix and contracting via sparse way.

-----

We now consider a small example, that with MPS Machine Learning. The virtual bond is set 3 and length is set 20. 

![benchmark](Batch TensorNetwork.assets/benchmark.png)

Notice:

- usually the `einsum` is provided for modern mathematic package.
- the contraction engine for `loop`, `batch contraction` is `tn.contractor.auto` which will calculate a 'optimized' path first (for uniform matrix chain, the path is just from left to right or verses). The `efficient coding` use assigned path: right to left path. The `way_vectorized_map` is parallel version for `loop`. 
- Replicated the result by `python speed_benchmark.py`
