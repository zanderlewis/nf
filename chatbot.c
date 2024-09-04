#include "nf.h"

int main() {
    TextRNN *rnn = create_text_rnn(HIDDEN_SIZE);
    
    // Read training data from file
    FILE *file = fopen("training_data.txt", "r");
    if (!file) {
        fprintf(stderr, "Failed to open training data file\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *training_data = malloc(file_size + 1);
    fread(training_data, 1, file_size, file);
    training_data[file_size] = '\0';
    fclose(file);

    train_text_rnn(rnn, training_data, EPOCHS_TG, LEARNING_RATE_TG);
    free(training_data);

    char input[100];
    printf("Enter a topic: ");
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = '\0';

    char *generated_text = generate_text(rnn, input, 100);
    printf("Generated text: %s\n", generated_text);

    free(generated_text);
    free_text_rnn(rnn);
    return 0;
}