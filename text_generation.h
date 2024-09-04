#ifndef TEXT_GENERATION_H
#define TEXT_GENERATION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "helpers.h"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    float *Wxh, *Whh, *Why, *bh, *by, *h;
    int epochs;
    float learning_rate;
} TextRNN;

#define VOCAB_SIZE 256
#define PRINT_INTERVAL_TG 1
#define EPOCHS_TG 10
#define LEARNING_RATE_TG 0.1

TextRNN* create_text_rnn(int hidden_size) {
    TextRNN *rnn = (TextRNN *)malloc(sizeof(TextRNN));
    if (!rnn) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    rnn->input_size = VOCAB_SIZE;
    rnn->hidden_size = hidden_size;
    rnn->output_size = VOCAB_SIZE;
    
    rnn->Wxh = create_embedding_chatbot(rnn->input_size * rnn->hidden_size);
    rnn->Whh = create_embedding_chatbot(rnn->hidden_size * rnn->hidden_size);
    rnn->Why = create_embedding_chatbot(rnn->hidden_size * rnn->output_size);
    rnn->bh = create_embedding_chatbot(rnn->hidden_size);
    rnn->by = create_embedding_chatbot(rnn->output_size);
    rnn->h = create_embedding_chatbot(rnn->hidden_size);
    rnn->epochs = EPOCHS_TG;
    rnn->learning_rate = LEARNING_RATE_TG;
    
    return rnn;
}

void free_text_rnn(TextRNN *rnn) {
    free(rnn->Wxh);
    free(rnn->Whh);
    free(rnn->Why);
    free(rnn->bh);
    free(rnn->by);
    free(rnn->h);
    free(rnn);
}

void train_text_rnn(TextRNN *rnn, const char *text, int epochs, float learning_rate) {
    int text_length = strlen(text);
    float *inputs = (float *)calloc(rnn->input_size, sizeof(float));
    float *targets = (float *)calloc(rnn->output_size, sizeof(float));
    float *h_next = (float *)calloc(rnn->hidden_size, sizeof(float));

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0;
        
        for (int t = 0; t < text_length - 1; t++) {
            // Prepare input and target
            memset(inputs, 0, rnn->input_size * sizeof(float));
            memset(targets, 0, rnn->output_size * sizeof(float));
            inputs[(unsigned char)text[t]] = 1;
            targets[(unsigned char)text[t+1]] = 1;
            
            // Forward pass
            for (int i = 0; i < rnn->hidden_size; i++) {
                h_next[i] = 0;
                for (int j = 0; j < rnn->input_size; j++) {
                    h_next[i] += rnn->Wxh[i * rnn->input_size + j] * inputs[j];
                }
                for (int j = 0; j < rnn->hidden_size; j++) {
                    h_next[i] += rnn->Whh[i * rnn->hidden_size + j] * rnn->h[j];
                }
                h_next[i] = sigmoid(h_next[i] + rnn->bh[i]);
            }
            
            float *output = (float *)calloc(rnn->output_size, sizeof(float));
            for (int i = 0; i < rnn->output_size; i++) {
                for (int j = 0; j < rnn->hidden_size; j++) {
                    output[i] += rnn->Why[i * rnn->hidden_size + j] * h_next[j];
                }
                output[i] = sigmoid(output[i] + rnn->by[i]);
            }
            
            // Compute loss
            for (int i = 0; i < rnn->output_size; i++) {
                total_loss += -targets[i] * log(output[i] + 1e-15) - (1 - targets[i]) * log(1 - output[i] + 1e-15);
            }
            
            // Backward pass (simplified, without full backpropagation through time)
            for (int i = 0; i < rnn->output_size; i++) {
                float d_output = output[i] - targets[i];
                for (int j = 0; j < rnn->hidden_size; j++) {
                    rnn->Why[i * rnn->hidden_size + j] -= learning_rate * d_output * h_next[j];
                }
                rnn->by[i] -= learning_rate * d_output;
            }
            
            // Update hidden state
            memcpy(rnn->h, h_next, rnn->hidden_size * sizeof(float));
            
            free(output);
        }
        
        printf("\033[1;33mEpoch %d, Loss: %f\033[0m\n", epoch, total_loss / text_length);
    }
    
    free(inputs);
    free(targets);
    free(h_next);
}

char* generate_text(TextRNN *rnn, const char *seed, int length) {
    char *generated_text = (char *)malloc((length + 1) * sizeof(char));
    float *inputs = (float *)calloc(rnn->input_size, sizeof(float));
    float *h = (float *)calloc(rnn->hidden_size, sizeof(float));
    
    // Initialize with seed
    int seed_length = strlen(seed);
    strncpy(generated_text, seed, seed_length);
    
    // Generate new text
    for (int i = seed_length; i < length; i++) {
        // Prepare input (use the last character of the current text)
        memset(inputs, 0, rnn->input_size * sizeof(float));
        inputs[(unsigned char)generated_text[i-1]] = 1;
        
        // Forward pass
        for (int j = 0; j < rnn->hidden_size; j++) {
            h[j] = 0;
            for (int k = 0; k < rnn->input_size; k++) {
                h[j] += rnn->Wxh[j * rnn->input_size + k] * inputs[k];
            }
            for (int k = 0; k < rnn->hidden_size; k++) {
                h[j] += rnn->Whh[j * rnn->hidden_size + k] * rnn->h[k];
            }
            h[j] = sigmoid(h[j] + rnn->bh[j]);
        }
        
        float *output = (float *)calloc(rnn->output_size, sizeof(float));
        for (int j = 0; j < rnn->output_size; j++) {
            for (int k = 0; k < rnn->hidden_size; k++) {
                output[j] += rnn->Why[j * rnn->hidden_size + k] * h[k];
            }
            output[j] = sigmoid(output[j] + rnn->by[j]);
        }
        
        // Sample from output distribution
        float sum = 0;
        for (int j = 0; j < rnn->output_size; j++) {
            sum += output[j];
        }
        float r = ((float)rand() / RAND_MAX) * sum;
        int sampled_char = 0;
        for (int j = 0; j < rnn->output_size; j++) {
            r -= output[j];
            if (r <= 0) {
                sampled_char = j;
                break;
            }
        }
        
        generated_text[i] = (char)sampled_char;
        
        // Update hidden state
        memcpy(rnn->h, h, rnn->hidden_size * sizeof(float));
        
        free(output);
    }
    
    generated_text[length] = '\0';
    
    free(inputs);
    free(h);
    
    return generated_text;
}

void save_text_rnn(TextRNN *rnn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for saving\n");
        return;
    }
    
    fwrite(&rnn->input_size, sizeof(int), 1, file);
    fwrite(&rnn->hidden_size, sizeof(int), 1, file);
    fwrite(&rnn->output_size, sizeof(int), 1, file);
    
    fwrite(rnn->Wxh, sizeof(float), rnn->input_size * rnn->hidden_size, file);
    fwrite(rnn->Whh, sizeof(float), rnn->hidden_size * rnn->hidden_size, file);
    fwrite(rnn->Why, sizeof(float), rnn->hidden_size * rnn->output_size, file);
    fwrite(rnn->bh, sizeof(float), rnn->hidden_size, file);
    fwrite(rnn->by, sizeof(float), rnn->output_size, file);
    
    fclose(file);
}

TextRNN* load_text_rnn(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for loading\n");
        return NULL;
    }
    
    TextRNN *rnn = (TextRNN *)malloc(sizeof(TextRNN));
    if (!rnn) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    fread(&rnn->input_size, sizeof(int), 1, file);
    fread(&rnn->hidden_size, sizeof(int), 1, file);
    fread(&rnn->output_size, sizeof(int), 1, file);
    
    rnn->Wxh = create_embedding_chatbot(rnn->input_size * rnn->hidden_size);
    rnn->Whh = create_embedding_chatbot(rnn->hidden_size * rnn->hidden_size);
    rnn->Why = create_embedding_chatbot(rnn->hidden_size * rnn->output_size);
    rnn->bh = create_embedding_chatbot(rnn->hidden_size);
    rnn->by = create_embedding_chatbot(rnn->output_size);
    rnn->h = create_embedding_chatbot(rnn->hidden_size);
    
    fread(rnn->Wxh, sizeof(float), rnn->input_size * rnn->hidden_size, file);
    fread(rnn->Whh, sizeof(float), rnn->hidden_size * rnn->hidden_size, file);
    fread(rnn->Why, sizeof(float), rnn->hidden_size * rnn->output_size, file);
    fread(rnn->bh, sizeof(float), rnn->hidden_size, file);
    fread(rnn->by, sizeof(float), rnn->output_size, file);
    
    fclose(file);
    return rnn;
}

#endif