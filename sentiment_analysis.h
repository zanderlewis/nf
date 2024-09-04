#ifndef SENTIMENT_ANALYSIS_H
#define SENTIMENT_ANALYSIS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "helpers.h"

#define MAX_WORDS 1000
#define MAX_WORD_LENGTH 50
#define HIDDEN_SIZE 64
#define EPOCHS 3000
#define LEARNING_RATE 0.01
#define PRINT_INTERVAL 500

typedef struct {
    char word[MAX_WORD_LENGTH];
    float *embedding;
} Word;

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    float *w1, *w2, *b1, *b2;
    int epochs;
    float learning_rate;
} NeuralNetwork;

Word vocabulary[MAX_WORDS];
int vocab_size = 0;

float* create_embedding(int size) {
    float *embedding = (float *)calloc(size, sizeof(float));
    if (!embedding) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        embedding[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    return embedding;
}

void add_word(const char *word) {
    if (vocab_size < MAX_WORDS) {
        strncpy(vocabulary[vocab_size].word, word, MAX_WORD_LENGTH - 1);
        vocabulary[vocab_size].word[MAX_WORD_LENGTH - 1] = '\0';
        vocabulary[vocab_size].embedding = create_embedding(HIDDEN_SIZE);
        vocab_size++;
    }
}

NeuralNetwork* create_nn(int input_size, int hidden_size, int output_size, int epochs, float learning_rate) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    nn->epochs = (epochs > 0) ? epochs : EPOCHS;  // Use default if not provided
    nn->learning_rate = (learning_rate > 0) ? learning_rate : LEARNING_RATE;  // Use default if not provided
    
    nn->w1 = create_embedding(input_size * hidden_size);
    nn->w2 = create_embedding(hidden_size * output_size);
    nn->b1 = create_embedding(hidden_size);
    nn->b2 = create_embedding(output_size);
    
    return nn;
}

void forward(NeuralNetwork *nn, float *input, float *hidden, float *output) {
    for (int i = 0; i < nn->hidden_size; i++) {
        hidden[i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            hidden[i] += input[j] * nn->w1[i * nn->input_size + j];
        }
        hidden[i] = sigmoid(hidden[i] + nn->b1[i]);
    }
    
    for (int i = 0; i < nn->output_size; i++) {
        output[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            output[i] += hidden[j] * nn->w2[i * nn->hidden_size + j];
        }
        output[i] = sigmoid(output[i] + nn->b2[i]);
    }
}

void train(NeuralNetwork *nn, float *input, float target) {
    float hidden[HIDDEN_SIZE];
    float output[1];
    float d_hidden[HIDDEN_SIZE];
    
    forward(nn, input, hidden, output);
    
    float error = target - output[0];
    float d_output = error * output[0] * (1 - output[0]);
    
    for (int j = 0; j < nn->hidden_size; j++) {
        nn->w2[j] += nn->learning_rate * d_output * hidden[j];
        d_hidden[j] = nn->w2[j] * d_output;
    }
    nn->b2[0] += nn->learning_rate * d_output;
    
    for (int i = 0; i < nn->hidden_size; i++) {
        float d_h = d_hidden[i] * hidden[i] * (1 - hidden[i]);
        for (int j = 0; j < nn->input_size; j++) {
            nn->w1[i * nn->input_size + j] += nn->learning_rate * d_h * input[j];
        }
        nn->b1[i] += nn->learning_rate * d_h;
    }
}

float* text_to_input(const char *text) {
    float *input = (float *)calloc(HIDDEN_SIZE, sizeof(float));
    if (!input) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    char *text_copy = strdup(text);
    if (!text_copy) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    char *word = strtok(text_copy, " ");
    int word_count = 0;
    
    while (word != NULL) {
        for (int i = 0; i < vocab_size; i++) {
            if (strcmp(word, vocabulary[i].word) == 0) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    input[j] += vocabulary[i].embedding[j];
                }
                word_count++;
                break;
            }
        }
        word = strtok(NULL, " ");
    }
    
    if (word_count > 0) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            input[i] /= word_count;
        }
    }
    
    free(text_copy);
    return input;
}

void train_nn(NeuralNetwork *nn, const char *positive_samples[], const char *negative_samples[], int num_samples) {
    for (int epoch = 0; epoch < nn->epochs; epoch++) {
        float total_error = 0;
        for (int i = 0; i < num_samples; i++) {
            float *pos_input = text_to_input(positive_samples[i]);
            float *neg_input = text_to_input(negative_samples[i]);
            
            float pos_output[1], neg_output[1];
            float hidden[HIDDEN_SIZE];
            
            forward(nn, pos_input, hidden, pos_output);
            total_error += fabs(1 - pos_output[0]);
            train(nn, pos_input, 1.0);
            
            forward(nn, neg_input, hidden, neg_output);
            total_error += fabs(0 - neg_output[0]);
            train(nn, neg_input, 0.0);
            
            free(pos_input);
            free(neg_input);
        }
        
        if (epoch % PRINT_INTERVAL == 0 || epoch == nn->epochs - 1) {
            printf("\033[1;37mEpoch %d, Average Error: %f\033[0m\n", epoch, total_error / (2 * num_samples));
        }
    }
}

void free_nn(NeuralNetwork *nn) {
    for (int i = 0; i < vocab_size; i++) {
        free(vocabulary[i].embedding);
    }
    free(nn->w1);
    free(nn->w2);
    free(nn->b1);
    free(nn->b2);
    free(nn);
}

float predict(NeuralNetwork *nn, const char *text) {
    float *input = text_to_input(text);
    float hidden[HIDDEN_SIZE];
    float output[1];
    
    forward(nn, input, hidden, output);
    
    return output[0];
}

void add_words(const char *words[], int count) {
    for (int i = 0; i < count; i++) {
        add_word(words[i]);
    }
}

void save_nn(NeuralNetwork *nn, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for saving\n");
        return;
    }

    // Write vocabulary size and words
    fwrite(&vocab_size, sizeof(int), 1, file);
    for (int i = 0; i < vocab_size; i++) {
        fwrite(vocabulary[i].word, sizeof(char), MAX_WORD_LENGTH, file);
        fwrite(vocabulary[i].embedding, sizeof(float), HIDDEN_SIZE, file);
    }

    // Write network structure and hyperparameters
    fwrite(&nn->input_size, sizeof(int), 1, file);
    fwrite(&nn->hidden_size, sizeof(int), 1, file);
    fwrite(&nn->output_size, sizeof(int), 1, file);
    fwrite(&nn->epochs, sizeof(int), 1, file);
    fwrite(&nn->learning_rate, sizeof(float), 1, file);

    // Write weights and biases
    fwrite(nn->w1, sizeof(float), nn->input_size * nn->hidden_size, file);
    fwrite(nn->w2, sizeof(float), nn->hidden_size * nn->output_size, file);
    fwrite(nn->b1, sizeof(float), nn->hidden_size, file);
    fwrite(nn->b2, sizeof(float), nn->output_size, file);

    fclose(file);
}

NeuralNetwork* load_nn(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file for loading\n");
        return NULL;
    }

    // Read vocabulary size and words
    fread(&vocab_size, sizeof(int), 1, file);
    for (int i = 0; i < vocab_size; i++) {
        fread(vocabulary[i].word, sizeof(char), MAX_WORD_LENGTH, file);
        vocabulary[i].embedding = (float *)malloc(HIDDEN_SIZE * sizeof(float));
        fread(vocabulary[i].embedding, sizeof(float), HIDDEN_SIZE, file);
    }

    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // Read network structure and hyperparameters
    fread(&nn->input_size, sizeof(int), 1, file);
    fread(&nn->hidden_size, sizeof(int), 1, file);
    fread(&nn->output_size, sizeof(int), 1, file);
    fread(&nn->epochs, sizeof(int), 1, file);
    fread(&nn->learning_rate, sizeof(float), 1, file);

    // Allocate memory for weights and biases
    nn->w1 = (float *)malloc(nn->input_size * nn->hidden_size * sizeof(float));
    nn->w2 = (float *)malloc(nn->hidden_size * nn->output_size * sizeof(float));
    nn->b1 = (float *)malloc(nn->hidden_size * sizeof(float));
    nn->b2 = (float *)malloc(nn->output_size * sizeof(float));

    if (!nn->w1 || !nn->w2 || !nn->b1 || !nn->b2) {
        fprintf(stderr, "Memory allocation failed\n");
        free(nn->w1);
        free(nn->w2);
        free(nn->b1);
        free(nn->b2);
        free(nn);
        fclose(file);
        return NULL;
    }

    // Read weights and biases
    fread(nn->w1, sizeof(float), nn->input_size * nn->hidden_size, file);
    fread(nn->w2, sizeof(float), nn->hidden_size * nn->output_size, file);
    fread(nn->b1, sizeof(float), nn->hidden_size, file);
    fread(nn->b2, sizeof(float), nn->output_size, file);

    fclose(file);
    return nn;
}

void print_results(NeuralNetwork *nn, const char *test_samples[], int num_samples) {
    printf("\033[1;36m\nResults:\033[0m\n\n");
    for (int i = 0; i < num_samples; i++) {
        float sentiment = predict(nn, test_samples[i]);
        printf("\033[1;37mSample %i:\033[0m \033[32m%.2f (%s)\033[0m\n", 
            i, 
            sentiment, 
            sentiment > 0.5 ? "Positive" : "Negative");
    }
}

#endif