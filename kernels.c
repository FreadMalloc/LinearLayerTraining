#include "pmsis.h"

////////////////////////////////////////// SINGLE CORE ///////////////////////////////////////////////////

void forewardProp(uint8_t * X_in, float * Y_out , float * W_wg, float * B_bs, uint32_t in, uint32_t out){
    uint32_t i, j;
    float a, b;
    for(j=0; j < out; j++){         // MAX 4 CORES
        float acc = 0;
        for(i=0; i< in; i++){       // 1920 times
            a = W_wg[j*in + i];
            b = (float)X_in[i];
            acc += a * b;
        }
        Y_out[j] = acc + B_bs[j];

        printf("%f\n", Y_out[j]);
    }
}

void validateLayer(float * Y_out, float *reference, uint32_t out)
{
    uint8_t j;
    float comparison_precision = 0.005;
    uint8_t ERRORflag = 0;

    for(j=0; j<out; j++){
        if( (Y_out[j]-reference[j] > comparison_precision) || (Y_out[j]-reference[j] < -comparison_precision) ){
            ERRORflag = 1;
            //printf("input %d is %f, reference is %f, therefore they are DIFFERENT\n", j, Y_out[j], reference[j]);
        }
        else{
            //printf("input %d is %f, reference is %f, therefore they are EQUAL\n", j, Y_out[j], reference[j]);
        }
    }

    if(ERRORflag == 1){
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("!!!!!!!!!!!!! VALIDATION TEST FAILED, THE MODEL IS NOT INITIALIZED CORRECTLY !!!!!!!!!!!!!!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
}
 
float cost_func(float * Y_out, float * Y_ex, uint32_t out){
    uint32_t j;
    float cost = 0.0, distance = 0.0;
    for(j=0; j<out; j++){
        distance = Y_out[j]-Y_ex[j];
        cost += distance * distance;
    }
    return cost/(float)out;
}

void backpropagation(uint8_t * X_in,
                        float * Y_out,
                        float * Y_ex,
                        float * W_wg,
                        float * B_bs,
                        uint32_t in,
                        uint32_t out,
                        float lr){
    uint32_t i, j, index;
    float gradient[out];
    //printf("%d,%d\n",in,out);

    // BIASES FIRST
    for(j=0; j< out; j++){
        gradient[j] = Y_out[j]-Y_ex[j];
        //float a = B_bs[j];   //to print
        B_bs[j] = B_bs[j] - lr * gradient[j];       // new bias = old - learning rate * (output - ideal output)
        //printf("bias %d was %f, since his y was off by %f, now it is %f\n", j, a, gradient[j], B_bs[j]);
    }

    // NOW WEIGHTS
    for(i=0; i< in; i++){
        for(j=0; j< out; j++){
            index = j*in + i;
            //float a = W_wg[index];   //to print
            
            W_wg[index] = W_wg[index] - lr * gradient[j] * (float)X_in[i];    // new weight = old - learning rate * (output - ideal output) * input
            
            //printf("weight %d was %f, since his y was off by %f and its input is %f, it has changed of %f\n", index, a, gradient[j], (float)X_in[i], W_wg[index]-a);
            //printf("(%d,%d), absolute: %d\n", j, i, index);
        }
    }

}


///////////////////////////////////////////// MULTI-CORE //////////////////////////////////////////////////

