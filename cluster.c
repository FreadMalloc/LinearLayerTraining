#include "pmsis.h"
#include "perf.h"
#include "rand_init.h"

#define IN 1920
#define OUT 4
#define WH IN*OUT


// APPLICATION DATA
PI_L2 uint8_t X_input[IN] = INPUTS_UINT8;
PI_L2 float Y_output[OUT];
PI_L2 float Verification_Vec[OUT] = VERIFICATION_FLOAT32;
PI_L2 float W_weight[WH] = WEIGHTS_FLOAT32;
PI_L2 float B_bias[OUT] = BIASES_FLOAT32;
float Y_expected[OUT] = {0.0, 1.0, 0.0, 0.0};
float learningRate = 0.00000001;

#if NUM_CORES == 1
  void forewardProp(uint8_t * X_in, float * Y_out , float * W_wg, float * B_bs, uint32_t in, uint32_t out);
  void validateLayer(float * Y_out, float *reference, uint32_t out);
  float cost_func(float * Y_out, float * Y_ex, uint32_t out);
  void backpropagation(uint8_t * X_in, float * Y_out, float * Y_ex, float * W_wg, float * B_bs, uint32_t in, uint32_t out, float lr);
#else
  //void vectAddPar(int * pSrcA, int  * pSrcB, int * pDstC, int n);
  //void vectAddParBalanced(int * pSrcA, int  * pSrcB, int * pDstC, int n);
#endif

/*
void vect_init(int * A, int * B, int * C, int size) {
  for (int i = 0; i < size; i++) {
    A[i] = i;
    B[i] = i+2;
    C[i] = 0;
  }
}
*/


void cluster_fn() {

  // init performance counters
  INIT_STATS();

  // executing the code multiple times to perform average statistics
  ENTER_STATS_LOOP();

  // start measuring
  START_STATS();

  // workload
  #if NUM_CORES == 1

    float cost = 0.0, new_cost = 0.0;
    
    forewardProp(X_input, Y_output, W_weight, B_bias, IN, OUT);
    validateLayer(Y_output, Verification_Vec, OUT);
    
    for(int epochs = 0; epochs<2; epochs++){
      cost = cost_func(Y_output, Y_expected, OUT);
      printf("cost: %f\n", cost);
      backpropagation(X_input, Y_output, Y_expected, W_weight, B_bias, IN, OUT, learningRate);
      forewardProp(X_input, Y_output, W_weight, B_bias, IN, OUT);
      new_cost = cost_func(Y_output, Y_expected, OUT);
      
      printf("new cost: %f\n     change: %f\n", new_cost, (cost-new_cost));
    }
  #else //multicore
  //  vectAddParBalanced(vectA, vectB, vectC, n);
  #endif

  // stop measuring
  STOP_STATS();

  // end of the performance statistics loop
  EXIT_STATS_LOOP();

#ifdef DEBUG  
  // check the result (optional)
//  vect_check(vectC, n);
#endif  
}
