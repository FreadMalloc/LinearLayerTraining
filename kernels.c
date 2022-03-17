#include "pmsis.h"

////////////////////////////////////////// SINGLE CORE ///////////////////////////////////////////////////

void forewardProp(uint8_t * X_in, float * Y_out , float * W_wg, float * B_bs, uint32_t in, uint32_t out){
    uint32_t i, j;
    float a, b, c, d, common;

    float acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    
    for(i=0; i< in; i++){       // 1920 times
        common = (float)X_in[i];

        a = W_wg[i];
        b = W_wg[in + i];
        c = W_wg[2*in + i];
        d = W_wg[3*in + i];

        acc0 += a * common;
        acc1 += b * common;
        acc2 += c * common;
        acc3 += d * common;
    }

    Y_out[0] = acc0 + B_bs[0];
    Y_out[1] = acc1 + B_bs[1];
    Y_out[2] = acc2 + B_bs[2];
    Y_out[3] = acc3 + B_bs[3];
    
    printf("%f\n%f\n%f\n%f\n", Y_out[0], Y_out[1], Y_out[2], Y_out[3]);
}

void validateLayer(float * Y_out, float *reference, uint32_t out)
{
    uint8_t j;
    float comparison_precision = 0.005;
    uint8_t ERRORflag = 0;

    for(j=0; j<out; j++){
        if( (Y_out[j]-reference[j] > comparison_precision) || (Y_out[j]-reference[j] < -comparison_precision) ){
            ERRORflag = 1;
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

    // BIASES FIRST
    for(j=0; j< out; j++){
        gradient[j] = Y_out[j]-Y_ex[j];
        B_bs[j] = B_bs[j] - lr * gradient[j];       // new bias = old - learning rate * (output - ideal output)
    }

    // NOW WEIGHTS
    for(i=0; i< in; i++){       //loop rolled on j

        float common_therm = X_in[i];
        //index is j*in + i
        W_wg[i]         = W_wg[i]        - lr * gradient[0] * common_therm;      // new weight = old - learning rate * (output - ideal output) * input
        W_wg[in + i]    = W_wg[in + i]   - lr * gradient[1] * common_therm;
        W_wg[2*in + i]  = W_wg[2*in + i] - lr * gradient[2] * common_therm;
        W_wg[3*in + i]  = W_wg[3*in + i] - lr * gradient[3] * common_therm;
    }

}


///////////////////////////////////////////// MULTI-CORE //////////////////////////////////////////////////

void forewardProp_Par(uint8_t * X_in, float * Y_out , float * W_wg, float * B_bs, uint32_t in, uint32_t out){
    uint32_t i, j;
    float a, b, c, d, common;

    float acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    
    for(i=0; i< in; i++){       // 1920 times
        common = (float)X_in[i];

        a = W_wg[i];
        b = W_wg[in + i];
        c = W_wg[2*in + i];
        d = W_wg[3*in + i];

        acc0 += a * common;
        acc1 += b * common;
        acc2 += c * common;
        acc3 += d * common;
    }

    Y_out[0] = acc0 + B_bs[0];
    Y_out[1] = acc1 + B_bs[1];
    Y_out[2] = acc2 + B_bs[2];
    Y_out[3] = acc3 + B_bs[3];
    
    printf("%f\n%f\n%f\n%f\n", Y_out[0], Y_out[1], Y_out[2], Y_out[3]);
}

void backpropagation_Par(uint8_t * X_in,
                        float * Y_out,
                        float * Y_ex,
                        float * W_wg,
                        float * B_bs,
                        uint32_t in,
                        uint32_t out,
                        float lr){
    uint32_t i, j, index;
    float gradient[out];

    // BIASES FIRST
    for(j=0; j< out; j++){
        gradient[j] = Y_out[j]-Y_ex[j];
        B_bs[j] = B_bs[j] - lr * gradient[j];       // new bias = old - learning rate * (output - ideal output)
    }

    // NOW WEIGHTS
    for(i=0; i< in; i++){       //loop rolled on j

        float common_therm = X_in[i];
        //index is j*in + i
        W_wg[i]         = W_wg[i]        - lr * gradient[0] * common_therm;      // new weight = old - learning rate * (output - ideal output) * input
        W_wg[in + i]    = W_wg[in + i]   - lr * gradient[1] * common_therm;
        W_wg[2*in + i]  = W_wg[2*in + i] - lr * gradient[2] * common_therm;
        W_wg[3*in + i]  = W_wg[3*in + i] - lr * gradient[3] * common_therm;
    }

}