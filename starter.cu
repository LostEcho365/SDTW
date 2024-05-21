#include <hip/hip_runtime.h>
#include <wb.h>
#include <dirent.h>

#define REFERENCE_LENGTH (100 * 1000)
#define QUERY_LENGTH (2 * 1000)
#define BATCH_SIZE 512
#define NORMALIZER_BLOCK_SIZE 1024

// For DTW
#define SEGMENT_SIZE 20 // Choose a divider of QUERY_LENGTH to avoid divergence
#define DTW_BLOCK_SIZE 512
// Minimize REFERENCE_LENGTH % (SEGMENT_SIZE * DTW_BLOCK_SIZE) for better utilization.
#define ITERATION_COUNT ceil((float) REFERENCE_LENGTH / (float) (SEGMENT_SIZE * DTW_BLOCK_SIZE))

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                               \
    if (err != hipSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got HIP error ...  ", hipGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

/**
 * Normalizer. Assuming 1 block for each string, to avoid reduction on the host or cross blocks.
 * The normalizer is used ONLY for query strings (512/batch x 2000 values).
 * For the query string, the output is the pointer to the entire batch, ie. 512 * 2000 elements.
 */
__global__ void normalizer(float *input, float *output, int length) {
  __shared__ float sharedSum[NORMALIZER_BLOCK_SIZE];
  __shared__ float sharedSquareSum[NORMALIZER_BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * length + tid;

  if (tid < length) {
    sharedSum[tid] = input[gid];
    sharedSquareSum[tid] = input[gid] * input[gid];
  } else {
    sharedSum[tid] = 0;
    sharedSquareSum[tid] = 0;
  }

  for (int i = tid + blockDim.x; i < length; i += blockDim.x) {
    int gi = blockIdx.x * length + i;
    sharedSum[tid] += input[gi];
    sharedSquareSum[tid] += input[gi] * input[gi];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      sharedSum[tid] += sharedSum[tid + s];
      sharedSquareSum[tid] += sharedSquareSum[tid + s];
    }
    __syncthreads();
  }

  float mean = sharedSum[0] / length;
  float stdDev = sqrt(sharedSquareSum[0] / length - mean * mean);

  for (int i = tid; i < length; i += blockDim.x) {
    int gi = blockIdx.x * length + i;
    output[gi] = (input[gi] - mean) / stdDev;
  }
}

__device__ int calculateReferenceOutputIndex(int intputIndex) {
  int iteration = intputIndex / (DTW_BLOCK_SIZE * SEGMENT_SIZE);
  int iterationPosition = intputIndex - iteration * (DTW_BLOCK_SIZE * SEGMENT_SIZE);
  int thread = iterationPosition / SEGMENT_SIZE;
  int cell = iterationPosition - thread * SEGMENT_SIZE;
  return iteration * (SEGMENT_SIZE * DTW_BLOCK_SIZE) + cell * DTW_BLOCK_SIZE + thread;
}

/**
 * It's copied from the above normalizer, except for the output index calculation.
 * The purpose of this method is to normalize the reference string, and store it in a way so the read is coalesced.
*/
__global__ void referenceNormalizer(float *input, float *output, int length) {
  __shared__ float sharedSum[NORMALIZER_BLOCK_SIZE];
  __shared__ float sharedSquareSum[NORMALIZER_BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * length + tid;

  if (tid < length) {
    sharedSum[tid] = input[gid];
    sharedSquareSum[tid] = input[gid] * input[gid];
  } else {
    sharedSum[tid] = 0;
    sharedSquareSum[tid] = 0;
  }

  for (int i = tid + blockDim.x; i < length; i += blockDim.x) {
    int gi = blockIdx.x * length + i;
    sharedSum[tid] += input[gi];
    sharedSquareSum[tid] += input[gi] * input[gi];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      sharedSum[tid] += sharedSum[tid + s];
      sharedSquareSum[tid] += sharedSquareSum[tid + s];
    }
    __syncthreads();
  }

  float mean = sharedSum[0] / length;
  float stdDev = sqrt(sharedSquareSum[0] / length - mean * mean);

  for (int i = tid; i < DTW_BLOCK_SIZE * SEGMENT_SIZE * ITERATION_COUNT; i += blockDim.x) {
    int inputGid = blockIdx.x * length + i;
    int outputGid = calculateReferenceOutputIndex(inputGid);

    if (inputGid < length) {
      output[outputGid] = (input[inputGid] - mean) / stdDev;
    } else {
      // Pad remainder with INFINITY to reduce divergence later.
      output[outputGid] = INFINITY;
    }
  }
}

/**
 * This kernel only reorders the reference, but not doing any normalization. It's useful for testing the costs for un-normalized strings.
*/
__global__ void reorderReference(float *input, float *output, int length) {
  for (int i = threadIdx.x; i < DTW_BLOCK_SIZE * SEGMENT_SIZE * ITERATION_COUNT; i += blockDim.x) {
    int inputGid = blockIdx.x * length + i;
    int outputGid = calculateReferenceOutputIndex(inputGid);

    if (inputGid < length) {
      output[outputGid] = input[inputGid];
    } else {
      output[outputGid] = INFINITY;
    }
  }
}

__global__ void dtw(float *reference, float *query, bool *matches, float threshold) {
  // Optimize with warp shuffle. (With this implementation, warp shuffle is slower, as it requires smaller block size.)
  __shared__ float sharedCosts[DTW_BLOCK_SIZE][3]; // column 0 for previous, 1 for current, 2 for local min;
  __shared__ float sharedQuery[QUERY_LENGTH];
  __shared__ float lastColumn[QUERY_LENGTH + 1]; // lastColumn is offset by 1, the first element holds 0 for very first query value.

  float localRefs[SEGMENT_SIZE];
  // previousCycleCosts is offset by 1. Index 0 holds value from previous thread (or the last column).
  float previousCycleCosts[SEGMENT_SIZE + 1];

  int tid = threadIdx.x;

  sharedCosts[tid][0] = INFINITY;
  sharedCosts[tid][1] = INFINITY;
  sharedCosts[tid][2] = INFINITY;
  
  int queryStart = blockIdx.x * QUERY_LENGTH;
  for (int blockQueryId = tid; blockQueryId < QUERY_LENGTH; blockQueryId += SEGMENT_SIZE) {
    sharedQuery[blockQueryId] = query[queryStart + blockQueryId];
    lastColumn[blockQueryId + 1] = INFINITY;
  }
  lastColumn[0] = 0;

  __syncthreads();

  for (int iteration = 0; iteration < ITERATION_COUNT; ++iteration) {
    // Load ref segment from global memory
    int startRef = iteration * (SEGMENT_SIZE * DTW_BLOCK_SIZE);

    for (int cell = 0; cell < SEGMENT_SIZE; ++cell) {
      localRefs[cell] = reference[startRef + cell * DTW_BLOCK_SIZE + tid];
      previousCycleCosts[cell + 1] = 0;
    }
    previousCycleCosts[0] = 0;

    for (int cycle = 0; cycle < QUERY_LENGTH + DTW_BLOCK_SIZE - 1; ++cycle) {
      int queryId = cycle - tid;

      if (queryId >= 0 && queryId < QUERY_LENGTH) {
        previousCycleCosts[0] = tid == 0 ? lastColumn[queryId] : sharedCosts[tid - 1][0];
        float left = tid == 0 ? lastColumn[queryId + 1] : sharedCosts[tid - 1][1];

        for (int cell = 0; cell < SEGMENT_SIZE; ++cell) {
          float diff = sharedQuery[queryId] - localRefs[cell]; 
          float upLeft = previousCycleCosts[cell];
          
          // If on the first query value (regardless of the current thread), only use the diff^2.
          float currentCellCost = diff * diff + fminf(fminf(left, previousCycleCosts[cell + 1]), upLeft);
          previousCycleCosts[cell] = left;
          left = currentCellCost;
        }
        
        sharedCosts[tid][0] = previousCycleCosts[SEGMENT_SIZE];
        sharedCosts[tid][1] = left;

        previousCycleCosts[SEGMENT_SIZE] = left;

        // Persist last column of the iteration.
        if (tid == DTW_BLOCK_SIZE - 1) {
          lastColumn[queryId + 1] = left;
        }
      }

      __syncthreads();

      // Do reduction when all the query for the thread is finished, and pass the min cost to next thread.
      if (queryId == QUERY_LENGTH - 1) {
        // If at the first thread, read min cost from the last thread (of the previous iteration).
        int sharedCostIndex = tid == 0 ? DTW_BLOCK_SIZE - 1 : tid - 1;
        float localMin = sharedCosts[sharedCostIndex][2];
        for (int cell = 0; cell < SEGMENT_SIZE; ++cell) {
          localMin = fminf(localMin, previousCycleCosts[cell + 1]);
        }
        sharedCosts[tid][2] = localMin;
      }
    }

    __syncthreads();
  }

  matches[blockIdx.x] = sharedCosts[DTW_BLOCK_SIZE - 1][2] < threshold;
}

void cpuTest(float *input) {
  float sum = 0;
  for (int i = 0; i < REFERENCE_LENGTH; i++) {
    sum += input[i];
  }
  float mean = sum / REFERENCE_LENGTH;

  float squareSum = 0;
  for (int i = 0; i < REFERENCE_LENGTH; i++) {
    squareSum += (input[i] - mean) * (input[i] - mean);
  }
  float stdDev = sqrt(squareSum / REFERENCE_LENGTH);

  wbLog(TRACE, "CPU mean: ", mean);
  wbLog(TRACE, "CPU std dev: ", stdDev);
}

int main(int argc, char **argv) {
  hipEvent_t dtwStart, dtwStop, normalizerStart, normalizerStop;
  hipEventCreate(&dtwStart);
  hipEventCreate(&dtwStop);
  hipEventCreate(&normalizerStart);
  hipEventCreate(&normalizerStop);

  wbArg_t args = wbArg_read(argc, argv);
  char *testNumber = wbArg_getTestNumber(args);

  char testDirectory[50];
  sprintf(testDirectory, "test/%s", testNumber);
  DIR *testDir = opendir(testDirectory);
  struct dirent *entry;
  
  // Reference string processing
  wbTime_start(Generic, "Importing data and creating memory on host");
  char referenceFile[50];
  char queryFiles[10][50];
  int queryFileCount = 0;

  // Order of the query files not guaranteed.
  while ((entry = readdir(testDir)) != NULL) {
      char *fileName = entry->d_name;
      char filePath[50];
      if (strncmp(fileName, "query_", strlen("query_")) == 0) {
        sprintf(filePath, "%s/%s", testDirectory, fileName);
        strcpy(queryFiles[queryFileCount], filePath);
        queryFileCount++;
      } else if (strncmp(fileName, "reference", strlen("reference")) == 0) {
        sprintf(filePath, "%s/%s", testDirectory, fileName);
        strcpy(referenceFile, filePath);
      }
  }

  // Read reference and query files into memory.
  float *hostReference;
  int referenceLength;
  float **hostQueries = (float **)malloc(queryFileCount * sizeof(float *));

  wbLog(TRACE, "Reading reference string from: ", referenceFile);
  hostReference = (float *)wbImport(referenceFile, &referenceLength);
  wbLog(TRACE, "Reference length: ", referenceLength);
  wbLog(TRACE, "Reference first: ", hostReference[0], ", last: ", hostReference[REFERENCE_LENGTH - 1]);
  wbLog(TRACE, "Finished reading reference string.");

  wbLog(TRACE, "Start reading query strings.");
  for (int i = 0; i < queryFileCount; ++i) {
    int queryBatchSize;
    int queryLength;
    wbLog(TRACE, "Reading query string from: ", queryFiles[i]);
    hostQueries[i] = (float *)wbImport(queryFiles[i], &queryBatchSize, &queryLength);
    wbLog(TRACE, "Query batch size: ", queryBatchSize, ", length: ", queryLength);
  }
  wbLog(TRACE, "Finished reading query strings.");

  // normalize reference string
  wbLog(TRACE, "Start normalizing reference string.");
  float *deviceReference;
  float *deviceNormalizedReference;
  size_t referenceSize = referenceLength * sizeof(float);
  hipMalloc((void **) &deviceReference, referenceSize);
  hipMalloc((void **) &deviceNormalizedReference, DTW_BLOCK_SIZE * SEGMENT_SIZE * ITERATION_COUNT * sizeof(float));
  hipMemcpy(deviceReference, hostReference, referenceSize, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(referenceNormalizer, dim3(1, 1, 1), dim3(NORMALIZER_BLOCK_SIZE, 1, 1), 0, 0, deviceReference, deviceNormalizedReference, referenceLength);
  //hipLaunchKernelGGL(reorderReference, dim3(1, 1, 1), dim3(NORMALIZER_BLOCK_SIZE, 1, 1), 0, 0, deviceReference, deviceNormalizedReference, referenceLength);
  wbCheck(hipGetLastError());
  hipDeviceSynchronize();
  wbLog(TRACE, "Finished normalizing reference string.");

  // Normalize query strings
  wbLog(TRACE, "Start normalizing query strings.");
  
  float **deviceQueries = (float **)malloc(queryFileCount * sizeof(float *));
  float **deviceNormalizedQueries = (float **)malloc(queryFileCount * sizeof(float *));
  size_t querySize = BATCH_SIZE * QUERY_LENGTH * sizeof(float);
  for (int i = 0; i < queryFileCount; ++i) {
    hipMalloc((void **) &deviceQueries[i], querySize);
    hipMalloc((void **) &deviceNormalizedQueries[i], querySize);
    hipMemcpy(deviceQueries[i], hostQueries[i], querySize, hipMemcpyHostToDevice);
  }

  float milliseconds = 0;
  hipEventRecord(normalizerStart, 0);
  for (int i = 0; i < queryFileCount; ++i) {
    hipLaunchKernelGGL(normalizer, dim3(BATCH_SIZE, 1, 1), dim3(NORMALIZER_BLOCK_SIZE, 1, 1), 0, 0, deviceQueries[i], deviceNormalizedQueries[i], QUERY_LENGTH);
    wbCheck(hipGetLastError());
  }
  hipEventRecord(normalizerStop, 0);
  hipEventSynchronize(normalizerStop);
  hipEventElapsedTime(&milliseconds, normalizerStart, normalizerStop);
  float throughput = 1000.0 * (queryFileCount * BATCH_SIZE * QUERY_LENGTH) / milliseconds / 1e9;

  wbLog(TRACE, "Finished normalizing query strings.");
  wbLog(TRACE, "Normalization throughput: ", throughput, " Giga samples/second.");

  // Run DTW.
  wbLog(TRACE, "Start DTW.");
  bool **hostOutputs = (bool **)malloc(queryFileCount * sizeof(bool *));
  bool **deviceOutputs = (bool **)malloc(queryFileCount * sizeof(bool *));
  size_t outputSize = BATCH_SIZE * sizeof(bool);

  for (int i = 0; i < queryFileCount; ++i) {
    hipMalloc((void **) &deviceOutputs[i], outputSize);
    hostOutputs[i] = (bool *)malloc(outputSize);
  }

  // warm up for DTW.
  float threshold = 5.0;
  hipLaunchKernelGGL(dtw, dim3(BATCH_SIZE, 1, 1), dim3(DTW_BLOCK_SIZE, 1, 1), 0, 0, deviceNormalizedReference, deviceNormalizedQueries[0], deviceOutputs[0], threshold);
  hipLaunchKernelGGL(dtw, dim3(BATCH_SIZE, 1, 1), dim3(DTW_BLOCK_SIZE, 1, 1), 0, 0, deviceNormalizedReference, deviceNormalizedQueries[0], deviceOutputs[0], threshold);
  hipDeviceSynchronize();

  milliseconds = 0;
  hipEventRecord(dtwStart, 0);
  for (int i = 0; i < queryFileCount; ++i) {
    // When testing DTW, cannot use normalized values, as they have different mean and std dev. Also, use reorderReference to NOT normalize the reference.
    //hipLaunchKernelGGL(dtw, dim3(BATCH_SIZE, 1, 1), dim3(DTW_BLOCK_SIZE, 1, 1), 0, 0, deviceNormalizedReference, deviceQueries[i], deviceOutputs[i]);
    hipLaunchKernelGGL(dtw, dim3(BATCH_SIZE, 1, 1), dim3(DTW_BLOCK_SIZE, 1, 1), 0, 0, deviceNormalizedReference, deviceNormalizedQueries[i], deviceOutputs[i], threshold);
    
    wbCheck(hipGetLastError());
  }
  hipEventRecord(dtwStop, 0);
  hipEventSynchronize(dtwStop);
  hipEventElapsedTime(&milliseconds, dtwStart, dtwStop);
  throughput = 1000.0 * (queryFileCount * BATCH_SIZE * QUERY_LENGTH) / milliseconds / 1e9;

  wbLog(TRACE, "DTW throughput: ", throughput, " Giga samples/second.");

  // Transfer the results back
  for (int i = 0; i < queryFileCount; ++i) {
    hipMemcpy(hostOutputs[i], deviceOutputs[i], outputSize, hipMemcpyDeviceToHost);
  }

  printf("Printing results from query: %s\n", queryFiles[0]);
  for (int i = 0; i < BATCH_SIZE; ++i) {
    printf("Match?[%d]: %d\n", i, hostOutputs[0][i]);
  }
  wbLog(TRACE, "Finished DTW.");
  
  // clean up
  for (int i = 0; i < queryFileCount; ++i) {
    hipFree(deviceNormalizedQueries[i]);
    hipFree(deviceQueries[i]);
    hipFree(deviceOutputs[i]);

    free(hostQueries[i]);
    free(hostOutputs[i]);
  }

  free(hostReference);
  free(deviceNormalizedQueries);
  free(deviceQueries);
  free(hostQueries);
  free(hostOutputs);

  hipFree(deviceReference);
  hipFree(deviceNormalizedReference);
  hipFree(deviceOutputs);

  hipEventDestroy(dtwStart);
  hipEventDestroy(dtwStop);
  hipEventDestroy(normalizerStart);
  hipEventDestroy(normalizerStop);

  return 0;
}
