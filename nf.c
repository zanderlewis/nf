#include "nf.h"

int main(int argc, char* argv[]) {
    int epochs = 1000;
    float learning_rate = 0.1;

    if (argc > 1) {
        epochs = atoi(argv[1]);
        if (epochs < 1) {
            printf("Invalid number of epochs\n");
            return 1;
        }
    }
    if (argc > 2) {
        learning_rate = atof(argv[2]);
        if (learning_rate <= 0) {
            printf("Invalid learning rate\n");
            return 1;
        }
    }

    const char *words[] = {
        "good", "excellent", "amazing", "wonderful", "great", "exciting",
        "interesting", "happy", "enjoyable", "bad", "terrible", "awful",
        "poor", "boring", "sad", "waste", "uninteresting"
    };
    add_words(words, sizeof(words) / sizeof(words[0]));

    NeuralNetwork *nn = create_nn(HIDDEN_SIZE, HIDDEN_SIZE, 1, epochs, learning_rate);
    
    const char *positive_samples[] = {
        "good movie", "excellent performance", "amazing experience",
        "wonderful story", "great acting", "exciting plot",
        "interesting characters", "happy ending", "enjoyable film",
        "amazing performance", "wonderful experience", "great to watch",
        
    };
    const char *negative_samples[] = {
        "bad film", "terrible acting", "awful screenplay",
        "poor direction", "disappointing movie", "boring story",
        "sad ending", "waste of time", "uninteresting plot",
        "disgusting movie", "terrible experience", "awful film",
        "poor quality", "boring film", "humiliating film", "waste of money"
    };
    int num_samples = sizeof(positive_samples) / sizeof(positive_samples[0]);
    
    // Training loop
    train_nn(nn, positive_samples, negative_samples, num_samples);
    
    // Example predictions
    const char *test_samples[] = {
        "good movie", "bad acting", "excellent story", "terrible ending",
        "amazing performance", "disappointing film", "exciting and interesting",
        "boring and awful", "wonderful experience", "poor quality", "Mr. Goodman is a bad person",
        "I love you", "The Badlands is an amazing place", "You look weird"
    };
    int num_test_samples = sizeof(test_samples) / sizeof(test_samples[0]);
    
    print_results(nn, test_samples, num_test_samples);

    save_nn(nn, "nn.bin");
    
    // Free memory
    free_nn(nn);
    
    return 0;
}