Sequence of operation: 
1. Linear extent (for the clean case) 
2. Alpha vs p and q (vary p and q individually)

--FIX A VALUE OF {P, Q}--

3. Plot alpha vs W for individual values of {p, q} (Take W up to 100)
4. Plot IPR as a function of time (do this for any particular {p, q})
Depending on how IPR varies, fix a time (one time in the rise region, one in the saturation region) and then plot IPR as a function of W (0.5 s, 6 s)

Repeat for n = 6, 7, 8 Plot IPR_sat vs W
                       Plot alpha vs W

5. Same as above, for S (for size, fix a value of 'n', and then take FRACTIONS of that (each slice increases the generation number))

--18th September--
Linear extent - vary p, q, n
Repeat HPC code with a much higher W range. np.arange(5, 200, 5)
Then, plot alpha vs W and IPR_sat vs W for n = 6, 7, 8 (visualise different generations on the same plot)