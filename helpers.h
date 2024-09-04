#ifndef HELPERS_H
#define HELPERS_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float* create_embedding_chatbot(int size) {
    float *embedding = (float *)malloc(size * sizeof(float));
    if (!embedding) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return embedding;
}

#endif