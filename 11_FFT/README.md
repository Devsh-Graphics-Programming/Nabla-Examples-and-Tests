# For Users

This will be a walkthrough of how to set up the code to run an FFT and an explanation of most stuff found in the FFT library (all the structs detailed here are in the workgroup namespace). The first thing to clarify is that since we're using Cooley-Tukey, we ONLY perform FFTs on power-of-two (PoT for short) sized arrays. If your array isn't PoT-sized, make sure to pad the array in whichever way you see fit up to a power of two.

To run an FFT, you need to call the FFT struct's static `__call` method. You do this like so: 

`FFT<Inverse, ConstevalParameters>::__call<Accessor, SharedMemoryAccessor>(Accessor accessor, SharedMemoryAccessor sharedMemoryAccessor)`

* `Inverse` indicates whether you're running a forward or an inverse FFT
* `ConstevalParameters` is a struct created from three compile-time constants: `ElementsPerInvocationLog2`, `WorkgroupSizeLog2` and `Scalar`. `Scalar` is just the scalar type for the complex numbers involved, `WorkgroupSizeLog2` is self-explanatory, and `ElementsPerInvocationLog2` is the (log of) the number of elements of the array each thread is tasked with computing, with the total ElementsPerInvocation being the length `FFTLength` of the array to perform an FFT on (remember it must be PoT) divided by the workgroup size used. This makes both `ElementsPerInvocation` and `WorkgroupSize` be PoT.
IMPORTANT: You MUST launch kernel with a workgroup size of `ConstevalParameters::WorkgroupSize` 
* `Accessor` is an accessor to the array. It MUST provide the methods `void get(uint32_t index, inout complex_t<Scalar> value)`, `void set(uint32_t index, in complex_t<Scalar> value)`,
which are hopefully self-explanatory. Furthermore, if doing an FFT with `ElementsPerInvocationLog2 > 1`, it MUST also provide a `void memoryBarrier()` method. If not accessing any type of memory during the FFT, it can be a method that does nothing. Otherwise, it must do a barrier with `AcquireRelease` semantics, with proper semantics for the type of memory it accesses. This example uses an Accessor going straight to global memory, so it requires a memory barrier. For an example of an accessor that doesn't, see the `28_FFTBloom` example, where we use preloaded accessors.
* `SharedMemoryAccessor` is an accessor to a shared memory array of `uint32_t` that MUST be able to fit `WorkgroupSize` many complex elements (one per thread). When instantiating a `workgroup::fft::ConstevalParameters` struct, you can access its static member field `SharedMemoryDWORDs` that yields the amount of `uint32_t`s the shared memory array must be able to hold. Furthermore, it MUST provide the methods `void get(uint32_t index, inout uint32_t value)`, `void set(uint32_t index, in uint32_t value)` (again self-explanatory) and `void workgroupExecutionAndMemoryBarrier()` (you can make this be a `glsl::barrier()`). 

## Utils

### Figuring out compile-time parameters
CPP-side (and soon HLSL as well) we provide a `workgroup::fft::optimalFFTParameters(uint32_t maxWorkgroupSize, uint32_t inputArrayLength)` function, which yields possible values for `ElementsPerInvocationLog2` and `WorkgroupSizeLog2` you might want to use to instantiate a `ConstevalParameters` struct. By default, we prefer to use only 2 elements per invocation when possible, and only use more if $2 \cdot \text{maxWorkgroupSize} < \text{inputArrayLength}$. This is because using more elements per thread either results in more accesses to the array via the `Accessor` or, if using preloaded accessors, it results in lower occupancy. `inputArrayLength` can be arbitrary, but please do note that the parameters returned will be for running an FFT on an array of length `roundUpToPoT(inputArrayLength)` and YOU are responsible for padding your data up to that size. You are, of course, free to choose whatever parameters are better for your use case, this is just a default.

### Indexing
We made some decisions in the design of the FFT algorithm pertaining to load/store order. In particular we wanted to keep stores linear to minimize cache misses when writing the output of an FFT. As such, the output of the FFT is not in its normal order, nor in bitreversed order (which is the standard for Cooley-Tukey implementations). Instead, it's in what we will refer to Nabla order going forward. The result of an FFT (either forward or inverse, assuming the input is in its natural order) will be referred to as an NFFT (N for Nabla). This NFFT contains the same elements as the DFT (which is the properly-ordered result of an FFT) of the same signal, just in Nabla order. We provide a struct `FFTIndexingUtils<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2>` that automatically handles the math for you in case you want to go from one order to the other. It provides the following methods:

* `uint32_t getDFTIndex(uint32_t outputIdx)`: given an index $\text{outputIdx}$ into the NFFT, it yields its corresponding $\text{freqIdx}$ into the DFT, such that 

    $\text{DFT}[\text{freqIdx}] = \text{NFFT}[\text{outputIdx}]$
* `uint32_t getNablaIndex(uint32_t freqIdx)`: given an index $\text{freqIdx}$ into the DFT, it yields its corresponding $\text{outputIdx}$ into the NFFT, such that 

    $\text{DFT}[\text{freqIdx}] = \text{NFFT}[\text{outputIdx}]$. It's essentially just the inverse of the previous method.
* `uint32_t getDFTMirrorIndex(uint32_t freqIdx)`: A common operation you might encounter using FFTs (especially FFTs of real signals) is to get the mirror around the middle (Nyquist frequency) of a given frequency. Given an index $\text{freqIdx}$ into the DFT, it returns a $\text{mirrorIndex}$ which is the index of its mirrored frequency, which satisfies the equation 

    $\text{freqIdx} + \text{mirrorIndex} = 0 \mod \text{FFTLength}$. Two elements don't have proper mirrors and are fixed points of this function: the Zero (index $0$ in the DFT) and Nyquist (index $\frac {\text{FFTLength}} 2$ in the DFT) frequencies. 
* `uint32_t getNablaMirrorIndex(uint32_t outputIdx)`: Yields the same as above, but the input and output are given in Nabla order. This is not to say we mirror $\text{outputIdx}$ around the middle frequency of the Nabla-ordered array (that operation makes zero sense) but rather this function is just $\text{getNablaIndex} \circ \text{getDFTMirrorIndex} \circ \text{getDFTIndex}$. That is, get the corresponding index in the proper DFT order, mirror THAT index around Nyquist, then go back to Nabla order. 

For the next two functions in this struct, let's give an example of where you might need them first. Supposed you packed two real signals $x, y$ as $x + iy$ and did a single FFT to save compute. Now you might want to unpack them to get the FFTs of each signal. If you had the DFT in the right order, unpacking requires to have values $\text{DFT}[T]$ and $\text{DFT}[-T]$ to unpack the values for each FFT at those positions. Suppose also that you are using preloaded accessors, so the whole result of the FFT is
currently resident in registers for threads in a workgroup. Each element a thread is currently holding is associated with a 
unique $\text{globalElementIndex}$, and to unpack some value a thread needs to know both $\text{NFFT}[\text{globalElementIndex}]$ and $\text{NFFT}[\text{getNablaMirrorIndex}(\text{globalElementIndex})]$. 

So, usually what you'd want to do is iterate over every $\text{localElementIndex}$ 
(which is associated with a $\text{globalElementIndex}$), get its mirror and do an unpack operation (an example of this is done 
in the Bloom example but iterating only over some values because we only unpack one half of the DFT since it's conjugate
symmetric for real signals). To get said mirror, we do a workgroup shuffle: with a shared memory array A, each thread of thread ID $\text{threadID}$ in a workgroup writes an element at $\text{A}[\text{threadID}]$ and reads a value from $\text{A}[\text{otherThreadID}]$, where 
$\text{otherThreadID}$ is the ID of the thread holding the element $\text{NFFT}[\text{getNablaMirrorIndex}(\text{globalElementIndex})]$ (again, see
the Bloom example for an example of this). This works assuming that each workgroup shuffle is associated with the same 
$\text{localElementIndex}$ for every thread. The question now becomes, how does a thread know which value it has to send in this shuffle?

The functions `FFTIndexingUtils::getNablaMirrorLocalInfo(uint32_t globalElementIndex)` and `FFTIndexingUtils::getNablaMirrorGlobalInfo(uint32_t globalElementIndex)` handle this for you: given a $\text{globalElementIndex}$, `getNablaMirrorLocalInfo` returns a struct with a field `otherThreadID` (the one we will receive a value from in the shuffle) and a field `mirrorLocalIndex` which is the $\text{localElementIndex}$ *of the element we should write to the shared memory array*. 
`getNablaMirrorGlobalInfo` returns the same info but with a `mirrorGlobalIndex` instead, so instead of returning the $\text{localElementIndex}$ of the element we have to send, it returns its $\text{globalElementIndex}$. 

In case this is hard to follow (because frankly it might be since we're working in weird nonstandard orderings of the DFT) you can copy the template function we use to trade mirrors around in `fft_mirror_common.hlsl` in the Bloom example. 


# For Maintainers

Note: All bitreversals are done by considering numbers as having exactly $\log_2(\text{FFTLength})$ bits in their binary representation (unless otherwise specified)

## Bit ordering of the Nabla FFT
As you might know, the Cooley-Tukey FFT outputs the elements of a DFT in bit-reversed order. Our implementation uses the Cooley-Tukey algorithm, but the way threads swap their elements around before performing each butterfly of the FFT makes the order end up slightly different. To perform an FFT on a $2^n$-sized array, we launch $\text{WorkgroupSize}$ many threads (in a single workgroup), each in charge of computing $\text{ElementsPerInvocation}$ positions of the FFT, where both are powers of two that multiply to $2^n$.

Here's what happens for the FFT of a 16-element array using $\text{ElementsPerInvocation} = 2$, $\text{WorkgroupSize} = 8$ (these parameters are actually illegal since we require $\text{WorkgroupSize} \ge 32$ but it's good for showing what happens under the hood). Please note below I use "invocation" and "thread" interchangeably.

![Radix 2 FFT](https://github.com/user-attachments/assets/dd926394-3175-4820-aed1-b7ab21f09be9)

Here's how to read this diagram: Since we're working with 16 elements with 2 elements per invocation, each invocation essentially holds two elements on which it performs a butterfly at any given time. On the left we have the input array with its elements numbered from 0 to 15, and their 4-bit representations. Colours are assigned per thread: thread 0 is blue, thread 1 is green and so on. Each invocation of id $\text{threadID}$ initially holds two elements, those indexed $\text{threadID}$ and $\text{threadID} + \text{WorkgroupSize}$. The elements a thread holds are called $\text{lo}$ and $\text{hi}$, based on which index is higher. You can at all times (meaning, looking at each column, since each column can be considered a different "time" of the computation) in the diagram tell that for each thread, in each column, there are two elements with that thread's colour. The one closest to the top is $\text{lo}$, the other is $\text{hi}$ (we start counting from the top).

From the first column to the next, we perform a butterfly. After the butterfly, from the second column to the third, we need threads to swap elements so they can keep computing the FFT. The first swap is done for each thread by `xor`ing its $\text{threadID}$ with half the current $\text{stride}$, which is the distance between the indices of $\text{lo}$ and $\text{hi}$ (this $\text{stride}$ starts at $\text{WorkgroupSize}$ and recurses by dividing by 2 at each step until it gets to 2, when $\text{lo}$ and $\text{hi}$ are next to each other) to get the ID of the thread it should swap values with. Then the "lowest" thread (the one for which `threadID & stride` yields 0) trades its $\text{hi}$ value with the "highest" thread's $\text{lo}$ value (the one for which the same operation yields $\text{stride}$). So for example on the third column you can see that the two blue elements are at a stride of 4 apart, after blue (thread 0) exchanges its $\text{hi}$ value with pink's (thread 4 = `0 ^ 4`) $\text{lo}$ value. 

For each column after that, we skip writing the butterflies (assume there is an implicit butterfly, but we're not intersted in that) and just write the swaps, which is what we want to follow. You can see at the end of the diagram (in the last column) that each thread ends up holding two consecutive elements of the output, in order: thread 0 holds the first two, thread 1 the next two, and so on. 

Now here's the catch: these outputs ARE bitreversed, it's Cooley-Tukey after all. But each thread writes its outputs to the same index it got its inputs from! For example thread 1 started with elements from input positions 1 and 9, so it will write the values it holds at the end to those same positions. In the FFT diagram, however, what thread 1 holds are the outputs indexed 2 and 3 (counting from the top), which bitreversed (remember that the result of a Cooley-Tukey FFT ends up bitreversed, and this IS a diagram for one such FFT) become 4 and 12. Next to each element at the end we write "X holds Y" to indicate that the element of index Y of the DFT is saved at position X in the output NFFT array. For example for thread 1 here we read "0001 holds 0100" (meaning that the output array, at the position of index 1, holds the element of index 4 of the DFT) and "1001 holds 1100" (meaning that the output array, at the position of index 9, holds the element of index 12 of the DFT).

In this basic case it is kinda easy to spot with the naked eye that a function mapping an index `outIdx` in the output NFFT array to its corresponding `freqIdx` in the DFT can be obtained by simply bitreversing all the lower bits, leaving the MSB fixed. For example, applying this operation to 1001 yields 1100, which is the mapping from index 9 in the output array to index 12 in the DFT that we showcased for one of thread 1's elements. Let's give such a function a name, `F`, which is a function such that $\text{NFFT[outIdx] = DFT[}F(\text{outIdx})\text{]}$.

### Generalizing
Let's move to 4 Elements Per Invocation (in this case this means having 4 invocations for a 16 element FFT) to see if we notice a pattern. 

![Radix 4 FFT](https://github.com/user-attachments/assets/f74a8397-f691-4ea4-81f9-7b82cc42eddd)

Now each thread holds 4 elements, each 4 positions apart. Each thread is now responsible for computing two butterflies at each step of the FFT (more generally, for an arbitrary number of `ElementsPerThread`, each thread will be in charge of computing `ElementsPerThread/2` butterflies). This means that in this case each thread has two $\text{lo}$ values, each with its corresponding $\text{hi}$ (and again `ElementsPerThread/2` of each in the general case). A rule of thumb for recognizing in a diagram which is which at each step of the FFT for a particular thread is to go through a column from the top of the bottom visiting all nodes with that thread's colour. Whenever you see an unmarked node, you mark it as a $\text{lo}$ and, remembering that each column is associated with a particular $\text{stride}$, you mark the node $\text{stride}$ positions below as its corresponding $\text{hi}$. This would be much easier to visualize with a full FFT diagram so that's my bad, but I labeled at each step for a given colour whether each element is which lo or hi for that thread at that current step. Still, drawing out the whole diagram might help you see better which values are being traded (remember that at the end of the day this is a normal FFT diagram). Butterflies are done between a lo and a hi of the same number. 

The rules for trading are as before: After each butterfly, the element of index `i` is swapped with the element of index `i ^ (stride) / 2`, where $\text{stride}$ is the distance between corresponding $\text{lo}$s and $\text{hi}$s. PLEASE NOTE that whoever is a $\text{lo}$ or a $\text{hi}$ at a current step has nothing to do with how they were traded. For example, going from column 2 to column 3 you might notice that there is an arrow going from the blue thread's `h1` to an `l2`. The labels might make this more confusing, but this is NOT saying that the values traded were between what blue held as `h1` and `l2` in the second column. Rather, by the rule mentioned before, element `8` is being swapped for element `12` between these columns (since we're considering `stride = 4` for the first swap). The labels are just there to tell you which elements participate in the implicit butterfly between that column and the next.

Let's go back to the previous diagram for a second (the one with 2 elements per invocation). We have a list going from top to bottom in which each line reads "X holds Y". Let's go through only the "X" values in order from top to bottom: `0000`, then `1000`, then `0001`, then `1001`, so on. If you notice, the lower 3 bits correspond with the number of invocations, while the top bit is used to indicate which element it is locally to the thread. For example, `0|110` is the first (0) element of thread 6 (`110`), while `1|110` is the second element of the same thread. 

Something similar happens in this new diagram: from the top to the bottom, we can write the storage indices as `localIndex|threadID`, just that we now need 2 bits to write the local indices (since we now have 4 elements per invocation) and 2 bits to write the threadID (since now we have only 4 threads). Another key thing to notice is that (if you don't believe me do this for other possible elements per invocation and convince yourself) we advance by enumerating 2 consecutive elements of each thread. For example in this last diagram we visit the first two elements of the 0th thread (`00|00` and `01|00`) then the same elemnts of the first thread (`00|01` and `01|01`) and so on, and once we've donde this for every thread we count the next two elements for all threads. If we had more elements per thread, this pattern continues: 5 and 6, then 7 and 8, and so on. 

If instead we decided to go through only the "Y" values instead, well, that's easy: again, this IS a Cooley-Tukey FFT diagram, so these elements are just the bitreversed indices from top to bottom (go look at a Decimation in Frequency diagram if you don't believe me). 

Assuming what I said above is true, here's a good way of finding out a formula for mapping output NFFT indices to DFT indices. On one column, write all indices using the rule specified above. That would be, start with `0|0` and `1|0`. Then comes `0|1, 1|1, 0|1, 1|2` all the way up to `0|WorkgroupSize - 1, 1|WorkgroupSize - 1`. Then start again with `2|0, 3|0, 2|1, 3|1, ..., 2|WorkgroupSize - 1, 3|WorkgroupSize - 1`. And then again until you get to `ElementsPerInvocation - 2|WorkgroupSize - 1, ElementsPerInvocation - 1|WorkgroupSize - 1`. Remember to write each number `a|b` in their binary form, with `log2(ElementsPerInvocation)` bits for `a` and `log2(WorkgroupSize)` bits for `b`. On the other column, just write all indices from 0 to `FFTSize - 1` but bitreversed.

Now we want to match these to find a rule. Here's an idea: The first column maps an index `n` (the position in the column, counting from the top) in the range `[0, FFTSize - 1]` to an index in the output NFFT array, let's call this mapping `e` for enumeration. This mapping depends on both `log2(ElementsPerInvocation)` and `log2(WorkgroupSize)`. The second column is also a mapping of `n` to an index in the (correctly ordered) DFT, and in fact we know this mapping to be `n -> bitreverse(n)`. We're almost done! Matching the columns like we have been, we now know that on line `n` we will read "`e(n)` holds `bitreverse(n)`", which means that `NFFT[e(n)] = DFT[bitreverse(n)]`. Then, line `e^{-1}(n)` will read "`n` holds `bitreverse(e^{-1}(n))`".  

So now let's compute `e`! `e(n)` can be computed with a circular right shift by one position of the lower `N - E + 1` bits of `n`, where `N` is the total number of bits used to represent `n` (so the base-2 logarithm of the `FFTSize`) and `E` is the base-2 logarithm of `ElementsPerThread`. Here's the intuition for why this works: Each two consecutive elements (when the first is even and the second is odd) correspond to two elements for the same thread, which are $\text{WorkgroupSize}$ apart. So `e(n)` is moving the lowest bit of `n` to be the lowest bit of `a` in `a|b` (this makes it so that every two consecutive numbers `2k` and `2k + 1`, the former gets mapped to `k` and the latter gets mapped to `k + WorkgroupSize`), and the rest of the lower bits now represent `b`, the `threadID`. With `n` going from `0` all the way up to `2 * WorkgroupSize - 1`, this generates the first "loop" from `0|0, 1|0` to `0|WorkgroupSize - 1, 1|WorkgroupSize - 1`. When `n = 2 * WorkgroupSize`, a `1` gets set as `a`'s next to lowest bit, and we repeat the loop. And each time we repeat the loop we add `1` to `a`'s highest bits. 

We have then worked out 

$F(n) = \text{bitreverse}(e^{-1}(n))$. 

For the function `F` that satisfies $\text{NFFT[outIdx] = DFT[}F(\text{outIdx})\text{]}$ (remember both `e`, the bitreversal and `F` are parameterized on the log2 of both `ElementsPerInvocation` and $\text{WorkgroupSize}$).
We're done! We now have that `NFFT[outputIdx] = DFT[F(outputIdx)]` and similarly `DFT[freqIdx] = NFFT[F^{-1}(freqIdx)]`. 

In code this is computed slightly differently, notice that we can define the map `g` by making `g` do a circular bit shift left by one position of the higher `N - E + 1` bits of `n`. This induces the relationships $\text{bitreverse} \circ e^{-1} = g^{-1} \circ \text{bitreverse}$ and $e \circ \text{bitreverse} = \text{bitreverse} \circ g$ which are what's used in code to compute this (there is no particular reason for this, experimentally I found those before having a proof so they stay because they're equivalent and I don't want to fix what's not broken). In the math lingo this means `e` and `g` are conjugate via `bitreverse`.

`F` is called `FFTIndexingUtils::getDFTIndex` and detailed in the users section above.



## Unpacking Rule for packed real FFTs

Following https://kovleventer.com/blog/fft_real/, we get the following equations:

$\text{DFT}_x[T] = \frac 1 2 \left(\text{DFT}[T] + \text{DFT}[-T]^* \right) = \frac 1 2 \left(\text{NFFT}[F^{-1}(T)] + \text{NFFT}[F^{-1}(-T)]^*\right)$

(with the equation for $\text{DFT}_y[T]$ being similar). Which then lets us work out

$\text{NFFT}_x[T] = \text{DFT}_x[F(T)] = \frac 1 2 \left(\text{NFFT}[T] + \text{NFFT}[F^{-1}(-F(T))]^*\right)$

and again a similar expression for $\text{NFFT}_y[T]$.

Notice the expression $F^{-1}(-F(T))$. This is what guides the `FFTIndexingUtils::getNablaMirrorIndex` talked about in the users section.

## Zero and Nyquist locations

Being special elements (especially of real FFTs, but also in general) we want to know where they end up. Since both bitreversal and the `e` function just move bits around, it's straightforward that `NFFT[0] = DFT[0]`, and `NFFT[0]` obviously always happens as thread 0's first element (that with index 0). 

Thread 0 also always holds the Nyquist frequency as its second element (that for which the local index is 1). In fact, from the rule we had deduced earlier, line 1 will read `e(1) holds bitreverse(1)`. `bitreverse(1)` is a 1 in the MSB followed by all 0s, which is exactly the Nyquist position ($2^{N-1}$) while `e(1)` works out to be `1|0` which means it's the second (index 1) element of thread 0. Another way to see this last statement is to notice that in an FFT diagram, the last butterfly is always in charge of computing the Zero and Nyquist frequencies.

## Finding out which elements to keep when doing a real-only FFT

When doing DFT on a real array, its result is known to be "symmetric conjugate" in a sense. That is, $DFT[T] = DFT[-T]^*$ (again, see https://kovleventer.com/blog/fft_real/ if you don't get the notation). If we had an ordered DFT, we could just keep elements $0$ through Nyquist (remember Nyquist is element indexed $\frac N 2$, for $N$ the length of the array) since all other elements are just the conjugate of one of these. So after doing our FFT, we need to figure out how to get these first elements. 

Here's an observation: each thread holds `ElementsPerThread` elements, and for a thread of ID `threadID` these happen to be `NFFT[threadID], NFFT[threadID + WorkgroupSize], ..., NFFT[threadID + (ElementsPerThread - 1) * WorkgroupSize]`. So the positions in the output NFFT array are parameterized by `threadID + k * WorkgroupSize`, where $\;0 \le k < \text{ElementsPerThread}$. We call an element of a thread even if its index is obtained from an even value of `k` in the previous parameterization. Enumerating the even elements in order produces a bitreversed lower half of the DFT. That is, the sequence `NFFT[0], NFFT[1], ..., NFFT[WorkgroupSize - 1], NFFT[0 + 2 * WorkgroupSize], NFFT[1 + 2 * WorkgroupSize], ..., NFFT[(WorkgroupSize - 1) + 2 * WorkgroupSize], NFFT[0 + 4 * WorkgroupSize], ..., NFFT[(WorkgroupSize - 1) + (ElementsPerThread - 2) * WorkgroupSize]` turns out to be exactly the lower half of the DFT (elements `0` through `Nyquist - 1` = $\frac N 2 - 1$, bitreversed *taking the indices as N-1 bit numbers* (so not taking into account the MSB of `0` they would have as indices in the whole DFT). 

Consider the lower half of the DFT. These are all indexed by `0|0` through `0|Nyquist - 1` with the `0` before the `|` being a single bit. Then, we map `0|n` to `0|bitreverse(n)`, where it's an `N-1` bit bitreversal. Now let's try calling `F^{-1}(0|bitreverse(n))` to figure out where each one of these ends up in the NFFT. From the identities worked out when we found `F`, we use `F^{-1} = e ∘ bitreverse` (this time it's an `N` bit bitreversal). So `F^{-1}(0|bitreverse(n)) = e(n|0)` (after applying bitreversal). Remember `n` is an `N-1` bit number, `e` performs a circular right shift of the lower `N - E + 1` bits of its argument, and the log of the $\text{WorkgroupSize}$, `W`, is such that `W + E = N`. So we can also consider `e` performs a circular right shift of the lower `W + 1` bits of its argument. Now let's consider what happens as we increase `n` starting from 0. The first $\text{WorkgroupSize}$ elements only need `W` bits to be written, so for these elements `e(n|0) = n`. So far so good, since this gives all elements of the form `threadID + 0 * WorkgroupSize` with `threadID` running from 0 to $\text{WorkgroupSize}$. Now, what happens to the next $\text{WorkgroupSize}$ elements? They can be written as `1|n` where `n` are now only the lower `W` bits. So `e(1|n|0) = 10|n`!. So the next $\text{WorkgroupSize}$ elements are all elements of the form `threadID + 2 * WorkgroupSize`. I hope it's not hard to see how it follows forwards until we've covered all the lower half of the DFT, because I am NOT writing the whole inductive proof for this one :) (but to me this is formal enough to understand that the proof is solid).  

Since `0` and `Nyquist` are also both real, we can conveniently pack them into the `0th` element as `NFFT[0] + i * NFFT[WorkgroupSize]` (remember from above discussion `NFFT[WorkgroupSize] = DFT[Nyquist]`).

# Assertions with no proof 

## Mirror trading when packing / unpacking real FFTs on preloaded accessors / avoiding global reads

To explain how things work, I'm going to give an example that happens in the load before the last IFFT in the bloom example. Say we did the FFT 
of a real signal like detailed above and we stored the lower half of the DFT in that manner (storing all even indices). Later, 
we want to do the IFFT of that signal. To perform the IFFT, we need all threads in the workgroup to have the corresponding
value of the FFT. We are also using preloaded accessors, so all values of the FFT need to be resident in registers for all threads
involved. Since we know the order in which we stored the even values, we recover all even values for every thread by reading those
from memory. We now need to set each thread's odd values before running the FFT. Since the even values covered the lower half 
of the DFT, it stands to reason (and with the same proof) that odd values cover the upper half of the DFT. Also, since it's 
the DFT of a real signal, it satisfies the conjugate symmetric property detailed earlier, meaning odd values are the conjugate 
of some even value. So, once we have loaded all even values, we want to go over every thread's odd values in order, figure out 
which even value of which thread is its conjugate, and set it to that value. A possible process could be, for each odd `localElementIndex`:

1. Take the `globalElementIndex` corresponding to this thread's current `localElementIndex`. 
2. Get a `mirrorIndex = FFTIndexingUtils::getNablaMirrorIndex(globalElementIndex)`, which is an even element of some other thread (it must be so because it's the mirror of an odd element and we ascertained that odd elements ar in the top half of the DFT and even elements in the bottom)
3. Figure out which thread holds the element `NFFT[mirrorIndex]`, which is just the thread of ID given by the lower `W` bits of `mirrorIndex`. Let's say that other thread has ID `otherThreadID`.

Once we have this, we know which thread we must receive our conjugated value from so we can store our odd local element. But how does that thread know it should 
send us that value? In fact, just like we are expecting a value from some thread, there is also another thread that expects us to send a value.
So we must figure out which value we are expected to send, and to whom. It turns out that the value we must send is obtained by taking 
`mirrorIndex` and replacing its lower bits (those that gave the other thread's ID) with the current thread's ID. This is the 
element at a local element index given by the higher bits of `mirrorIndex`. This also means that trades always happen 1 on 1: 
The thread I send a value to is the same thread I will receive a value from. 

Haven't come up with a proof for this one :(