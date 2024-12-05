# Bit ordering of the Nabla FFT
As you might know, the Cooley-Tukey FFT outputs the elements of a DFT in bit-reversed order. Our implementation uses the Cooley-Tukey algorithm, but the way threads swap their elements around before performing each butterfly of the FFT makes the order end up slightly different. To perform an FFT on a $2^n$-sized array, we launch `WorkgroupSize` many threads (in a single workgroup), each in charge of computing `ElementsPerInvocation` positions of the FFT, where both are powers of two that multiply to $2^n$.

Here's what happens for the FFT of a 16-element array using `ElementsPerInvocation = 2`, `WorkgroupSize = 8`. Please note below I use "invocation" and "thread" interchangeably.

![Radix 2 FFT](https://github.com/user-attachments/assets/8ceacf0d-d615-4b5a-9e32-3b421c70846a)

Here's how to read this diagram: Since we're working with 16 elements with 2 elements per invocation, each invocation essentially holds two elements on which it performs a butterfly at any given time. On the left we have the input array with its elements numbered from 0 to 15, and their 4-bit representations. Colours are assigned per thread: thread 0 is blue, thread 1 is green and so on. Each invocation of id `threadID` initially holds two elements, those indexed `threadID` and `threadID + WorkgroupSize`. The elements a thread holds are called `lo` and `hi`, based on which index is higher. You can at all times in the diagram tell that for each thread, in each column, there are two elements with that thread's colour. The one closest to the top is `lo`, the other is `hi` (we start counting from the top).

From the first column to the next, we perform a butterfly. After the butterfly, from the second column to the third, we need threads to swap elements so they can keep computing the FFT. You can read this in the code, but basically the first swap is done for each thread by `xor`ing its `threadID` with half the current `stride`, which is the distance between the indices of `lo` and `hi` (this `stride` starts at `WorkgroupSize` and recurses by dividing by 2 at each step until it gets to 2, when `lo` and `hi` are next to each other) to get the ID of the thread it should swap values with. Then the "lowest" thread (the one for which `threadID & stride` yields 0) trades its `hi` value with the "highest" thread's `lo` value (the one for which the same operation yields `stride`). So for example on the third column you can see that the two blue elements are at a stride of 4 apart, after blue (thread 0) exchanges its `hi` value with pink's (thread 4 = `0 ^ 4`) `lo` value. 

For each column after that, we skip writing the butterflies (we're not intersted in that) and just write the swaps. You can see at the end that each thread ends up holding two consecutive elements of the output, in order: thread 0 holds the first two, thread 1 the next two, and so on. 

Now here's the catch: these outputs ARE bitreversed, it's Cooley-Tukey after all. But each thread writes its outputs to the same index it got its inputs from! So for example thread 1 grabbed elements from input positions 1 and 9, so it will write whatever it is holding to those same indices. But it holds outputs indexed 2 and 3 (counting from the top), which bitreversed become 4 and 12. So when writing to the output, this does not end up bitreversed. Next to each element at the end we write "X holds Y" to indicate that the element of index Y of the DFT is saved at position X in the output. 

The radix-2 FFT case is kinda easy to see with a naked eye that a function mapping an index `outIdx` in the output array to its corresponding `freqIdx` in the DFT can be obtained by simply bitreversing all the lower bits, leaving the MSB fixed. Let's give such a function a name. For reasons that will become clearer later, this function will be parameterized by the logarithm of `ElementsPerThread`. So in this case we have found an expression for $F_1(n)$, which is a function such that $\text{output[outIdx] = DFT[}F_1(outIdx)\text{]}$

## Generalizing
Let's move to 4 Elements Per Invocation (in this case this means having 4 invocations for a 16 element FFT) to see if we notice a pattern. 

![Radix 4 FFT](https://github.com/user-attachments/assets/2c14c695-764b-4910-b556-e6ce4ee68223)

Now each thread holds 4 elements, each 4 positions apart. Each thread is now responsible for computing two butterflies at each step of the FFT (more generally, for an arbitrary number of `ElementsPerThread`, each thread will be in charge of computing `ElementsPerThread/2` butterflies). This means that in this case each thread has two `lo` values, each with its corresponding `hi` (and again `ElementsPerThread/2` of each in the general case). A rule of thumb for recognizing in a diagram which is which at each step of the FFT for a particular thread is to go through a column from the top of the bottom visiting all nodes with that thread's colour. Whenever you see an unmarked node, you mark it as a `lo` and, remembering that each column is associated with a particular `stride`, you mark the node `stride` positions below as its corresponding `hi`. 
Note that from columns 2 to 3, after the first butterfly, no swaps *between threads* need to be done, since each thread will already hold the elements it needs to continue. However, what's going on under the hood is that each thread is swapping one of its own `lo` values with one of its `hi` values, so there is still a swap going on. Since the colours being all the same might make it confusing, to remember which element is being swapped with which remember the stride rule, or draw out the full FFT diagram (this is just a radix-2 FFT diagram with coloured nodes where we just didn't draw out all the butterflies). Once again, since it's still a normal radix-2 diagram, the output is still bitreversed, but now where each element is stored is different. It seems like we can spot a pattern though, and here it is (if you don't believe me try doing the diagram for 8 Elements Per Invocation, perhaps a 32 element FFT will be better for that):

Notice how the indices grew in the previous diagram: `0000`, then `1000`, then `0001`, then `1001`, so on. If you notice, the lower 3 bits correspond with the number of invocations, while the top bit is used to indicate which element it is locally to the thread. For example, `0|110` is the first (0) element of thread 6 (`110`), while `1|110` is the second element of the same thread. Something similar happens in this new diagram: from the top to the bottom, we can write the storage indices as `localIndex|threadID`. Another key thing to notice is that you'll get two consecutive elements of each thread, going through every thread and then looping again to the next two elements of the first thread and so on until you have gone through all the indices. 

You can convince yourself that this pattern holds for any amount of Elements Per Invocation, and that the way the pattern changes only depends on `ElementsPerInvocation`.

So assuming what I said above is true, here's a good way of finding out a formula for mapping output indices to DFT indices. On one column, write all indices using the rule specified above. That would be, start with `0|0` and `1|0`. Then comes `0|1, 1|1, 0|1, 1|2` all the way up to `0|WorkgroupSize - 1, 1|WorkgroupSize - 1`. Then start again with `2|0, 3|0, 2|1, 3|1, ..., 2|WorkgroupSize - 1, 3|WorkgroupSize - 1`. And then again until you get to `ElementsPerInvocation - 2|WorkgroupSize - 1, ElementsPerInvocation - 1|WorkgroupSize - 1`. Remember to write each number `a|b` in their binary form, with `log2(ElementsPerInvocation)` bits for `a` and `log2(WorkgroupSize)` bits for `b`. On the other column, just write all indices from 0 to `FFTSize - 1` but bitreversed. This is what we write at the end of the diagram! The part that says "X holds Y".

Now we want to match these to find a rule. Here's an idea: The first column maps a number (the position in the column, counting from the top) in the range `[0, FFTSize - 1]` to an index in the output array, let's call this mapping $e_{\log_2(\text{ElementsPerInvocation})}$: $e$ for enumerate, and to show that it is parameterized by the logarithm of `ElementsPerThread`. Let's just write `e` in the next part for simplicity. The second column is the mapping of a number in the same range to an index in the (correctly ordered) DFT, and in fact we know this mapping to be `freqIdx -> bitreverse(freqIdx)`. We're almost done! Matching the columns like we have been, we now know that on line `n` we will read "`e(n)` holds `bitreverse(n)`", which means that `output[e(n)] = DFT[bitreverse(n)]`. Then, line `e^{-1}(n)` will read "`n` holds `bitreverse(e^{-1}(n))`".  

So now let's compute `e`! I will assert without proof that `e(n)` can be computed with a circular right shift by one position of the lower `N - E + 1` bits of `n`, where `N` is the total number of bits used to represent `n` (so the base-2 logarithm of the `FFTSize`) and `E` is the base-2 logarithm of `ElementsPerThread`. Here's the intuition for why this works: Each two consecutive elements (when the first is even and the second is odd) correspond to two elements for the same thread. So `e(n)` is moving the lowest bit of `n` to be the lowest bit of `a` in `a|b`, and the rest of the lower bits now represent `b`, the `threadID`. With `n` going from `0` all the way up to `2 * WorkgroupSize - 1`, this generates the first "loop" from `0|0, 1|0` to `0|WorkgroupSize - 1, 1|WorkgroupSize - 1`. When `n = 2 * WorkgroupSize`, a `1` gets set as `a`'s next to lowest bit, and we repeat the loop. And each time we repeat the loop we add `1` to `a`'s highest bits. 

We have then worked out 

$F(n) = \text{bitreverse}(e^{-1}(n))$. 

Where the subindex for the log of `ElementsPerThread` is implicit for readability. We're done! We now have that `output[outputIdx] = DFT[F(outputIdx)]` and similarly `DFT[freqIdx] = output[F^{-1}(freqIdx)]` 

In code this is computed slightly differently, notice that we can define the map `g` by making `g` do a circular bit shift left by one position of the higher `N - E + 1` bits of `n`. This induces the relationships $\text{bitreverse} \circ e^{-1} = g^{-1} \circ \text{bitreverse}$ and $e \circ \text{bitreverse} = \text{bitreverse} \circ g$ which are what's used in code to compute this (for no particular reason, they were expressions we had deduced before we had a proof for the mapping). In the math lingo this means `e` and `g` are conjugate via `bitreverse`.



# Unpacking Rule for packed real FFTs

Following https://kovleventer.com/blog/fft_real/, we get the following equations:

$\text{DFT}_x[T] = \frac 1 2 \left(\text{DFT}[T] + \text{DFT}[-T]^\* \right) = \frac 1 2 \left(\text{output}[F^{-1}(T)] + \text{output}[F^{-1}(-T)]^\*\right)$

(with the equation for `DFT_y[T]` being similar). Which then lets us work out

$\text{output}_x[T] = \text{DFT}_x[F(T)] = \frac 1 2 \left(\text{output}[T] + \text{output}[F^{-1}(-F(T))]^\*\right)$

and again a similar expression for `output_y[T]`

# Nyquist location

Thread 0 always holds the Nyquist frequency as its second element. In fact, from the rule we had deduced earlier, line 1 will read `e(1) holds bitreverse(1)`. `bitreverse(1)` is a 1 in the MSB followed by all 0s, which is exactly the Nyquist position ($2^{N-1}$) while `e(1)` works out to be `1|0` which means it's the second (index 1) element of thread 0

# Experimental assertions with no proof

## Finding out which elements to keep when doing a real-only FFT

When doing DFT on a real array, its result is known to be "symmetric conjugate" in a sense. That is, $DFT[T] = DFT[-T]^*$ (again, see https://kovleventer.com/blog/fft_real/ if you don't get the notation). If we had an ordered DFT, we could just keep elements $0$ through Nyquist (remember Nyquist is element indexed $\frac N 2$, for $N$ the length of the array) since all other elements are just the conjugate of one of these. So after doing our FFT, we need to figure out how to get these first elements. 

Here's an observation: each thread holds `ElementsPerThread` elements, and for a thread of ID `threadID` these happen to be `output[threadID], output[threadID + WorkgroupSize], ..., output[threadID + (ElementsPerThread - 1) * WorkgroupSize]`. So the positions in the output array are parameterized by `threadID + k * WorkgroupSize`, where $0 \le k < \text{ElementsPerThread}$. We call an element of a thread even if its index is obtained from an even value of `k` in the previous parameterization. I will assert without proof here *based on experimental observations that seem to generalize* that storing the even elements produces a bitreversed lower half of the DFT. That is, the sequence `output[0], output[1], ..., output[WorkgroupSize - 1], output[0 + 2 * WorkgroupSize], output[1 + 2 * WorkgroupSize], ..., output[(WorkgroupSize - 1) + 2 * WorkgroupSize], output[0 + 4 * WorkgroupSize], ..., output[(WorkgroupSize - 1) + (ElementsPerThread - 2) * WorkgroupSize]` turns out to be exactly the lower half of the DFT (elements `0` through `Nyquist - 1` = $\frac N 2 - 1$, bitreversed *taking the indices as N-1 bit numbers* (so not taking into account the MSB of `0` they would have as indices in the whole DFT). Since `0` and `Nyquist` are also both real, we can conveniently pack them into the `0th` element as `output[0] + i * output[WorkgroupSize]` (remember from above discussion `output[WorkgroupSize] = DFT[Nyquist]`).
