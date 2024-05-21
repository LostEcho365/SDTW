# gpu-final-project

## User Guide
1. To build the project: `./build starter.cu <build_name>`
1. To run tests: `sbatch run_tests <build_name> <test_number>`
1. There are 2 binaries included, both prints the throughput of after processing all the batches.
    1. project_costs (outdated, lower performance): Prints the minimum cost of the FIRST batch. This binary is for testing and verification results.
    1. project: Prints if the query string is a match. 0 for false, 1 for true with hard coded threshold 5. And prints the throughput for all batches. The binary only prints the results of the FIRST batch (first batch is NOT necessarily query_0.raw if there multiple query batches). 
1. final_project.tar contains the source code for the project, a test generator and the test data.
1. There are 2 sets of test data:
    1. test/0 contains 10 random queries, and 1 random reference.
    1. test/1 contains 1 query, with index 0 and 2 replaced by segments from reference string.
1. The test datasets have the following formats:
    1. query_<batch_num>.raw: 1 batch of the query string. 512 x 2000 metrics. The first line is the dimension.
    1. reference.raw: reference string 100000 x 1. Each line is an element.

## Notes 03/05
1. Updated the reference/query dimensions according to https://edstem.org/us/courses/50708/discussion/4505494. As aresult, all the test data were invalidated. Regenerated test/0 and test/1
1. test/0 contains 10 random queries, and 1 random reference.
1. test/1 contains 1 query, with index 0 and 2 replaced by segments from reference string.
    1. index 0: offset by 50,000 from reference.
    1. index 2: offset by 20,000 from reference.
1. Uploaded 2 test results
    1. 27788: ran against un-normalized data
    1. 27810: ran against normalized data

## More notes
1. Have to rebuild the libwb, but deleting libwb/build and run `./build ...` at project root.
1. Added test_gen. Sharing the same `run_tests` script as the main program.
1. Introduced new option `-n`, which is the test number.
1. Updated `run_tests` to use the new options.

## Notes
1. About test data
    1. test/0
        1. contains 3 query batches. They were generated using the same seed, so high duplications. Better use /1
    1. test/1
        1. different seeds for reference and query_0
        1. query_0 took 2 segments (of size 2048) from reference, replaced string at index 0 and 2.
        1. For the first segment (at index 0), the offset is 50,000.
        1. For the second segment (at index 2), the offset is 20,000.
    1. all references are not normalized. This is more relevant for testing individual kernels.
1. About testing kernels
    1. When testing the DTW kernel, use the un-normalized data directly. Otherwise, the data will be normalized with different means and stddev, so they will not align perfectly (ie. cost == 0). However, the query string at index 0 and 2 (for test/1) should still be significantly lower than the other values (in my test results <1, while the others >500).
    1. If using the un-normalized data, the cost for these 2 query strings should be exactly 0.
1. About test results
    1. Included 2 test files, both for test/1 on a single batch of queries (512 queries, each 2048).
    1. slurm-26306.out -> un-normalized data
    1. slurm-26307.out -> normalized data
1. About performance
    1. tl;dr; not optimized too well (at all) :(
    1. **Important**: The code has assumption about the dimensions of kernel and input data. Changing those, including reference length, query length, batch size, segment size, will break the code... more details commented in code.

Special credits to Ryan Chen and James O'Donoghue for the initial work into this project
