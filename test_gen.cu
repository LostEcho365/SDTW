#include <hip/hip_runtime.h>
#include <wb.h>
#include <string.h>

#define REFERENCE_LENGTH (100 * 1000)
#define QUERY_LENGTH (2 * 1000)
#define BATCH_SIZE 512
#define VAL_MIN 58.0
#define VAL_MAX 120.0

float generateRandomValue() {
    float random = (float)rand() / (float)RAND_MAX;
    return VAL_MIN + random * (VAL_MAX - VAL_MIN);
}

/**
 * Populate random reference data.
*/
void populateReference(const char *testNumber) {
    char referenceLocation[50];
    sprintf(referenceLocation, "test/%s/reference.raw", testNumber);
    
    srand(time(NULL));
    float *testReference = (float *)malloc(REFERENCE_LENGTH * sizeof(float));
    for (int i = 0; i < REFERENCE_LENGTH; ++i) {
        testReference[i] = generateRandomValue();
    }

    wbExport(referenceLocation, testReference, REFERENCE_LENGTH);
    printf("Created reference file: %s\n", referenceLocation);
    free(testReference);
}

void populateQuery(const char *testNumber) {
    srand(time(NULL));
    float *testQuery = (float *)malloc(QUERY_LENGTH * BATCH_SIZE * sizeof(float));

    for (int i = 0; i < 10; ++i) {
        char queryLocation[50];
        sprintf(queryLocation, "test/%s/query_%d.raw", testNumber, i);

        for (int i = 0; i < QUERY_LENGTH * BATCH_SIZE; ++i) {
            testQuery[i] = generateRandomValue();
        }
        wbExport(queryLocation, testQuery, BATCH_SIZE, QUERY_LENGTH);
        printf("Created query file: %s\n", queryLocation);
    }

    free(testQuery);
}

// TODO: Currently not in use. Needs to be modified to work with the new setup.
void populateQueryWithReference(const char *file, float *reference) {
  srand(time(NULL));
  float *testQuery = (float *)malloc(QUERY_LENGTH * BATCH_SIZE * sizeof(float));

  for (int i = 0; i < QUERY_LENGTH * BATCH_SIZE; ++i) {
    testQuery[i] = generateRandomValue();
  }

  // Override bath 0 and batch 2
  for (int i = 0; i < QUERY_LENGTH; ++i) {
    testQuery[i] = reference[i + 50000];
    testQuery[QUERY_LENGTH * 2 + i] = reference[i + 20000];
  }

  wbExport(file, testQuery, BATCH_SIZE, QUERY_LENGTH);
  free(testQuery);
}

int main(int argc, char **argv) {
  wbArg_t args;

  // Reference string variables
  float *hostReference;
  int referenceLength;

  args = wbArg_read(argc, argv);

  // For creating test data
  char *testNumber = wbArg_getTestNumber(args);
  populateReference(testNumber);
  populateQuery(testNumber);
  
  // Reference string processing
  hostReference = (float *)wbImport("test/3/reference.raw", &referenceLength);
  //populateQueryWithReference("test/4/query_0.raw", hostReference);

  wbLog(TRACE, "length: ", referenceLength);
  wbLog(TRACE, "first: ", hostReference[0], ", last: ", hostReference[REFERENCE_LENGTH - 1]);

  return 0;
}
