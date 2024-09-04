#include "nf.h"

int main() {
    // Expanded vocabulary
    const char *words[] = {
        "good", "excellent", "amazing", "wonderful", "great", "exciting",
        "interesting", "happy", "enjoyable", "bad", "terrible", "awful",
        "poor", "boring", "sad", "waste", "uninteresting"
    };
    add_word_array(words, sizeof(words) / sizeof(words[0]));

    NeuralNetwork *nn = create_nn(HIDDEN_SIZE, HIDDEN_SIZE, 1);
    
    // Expanded training data
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
    int num_samples = 16;
    
    // Training loop
    train_nn(nn, positive_samples, negative_samples, num_samples);
    
    // Example predictions
    const char *test_samples[] = {
        "good movie", "bad acting", "excellent story", "terrible ending",
        "amazing performance", "disappointing film", "exciting and interesting",
        "boring and awful", "wonderful experience", "poor quality"
    };
    int num_test_samples = 10;
    
    for (int i = 0; i < num_test_samples; i++) {
        float sentiment = predict(nn, test_samples[i]);
        printf("Sentiment for '%s': %.2f (%s)\n", test_samples[i], sentiment, sentiment > 0.5 ? "Positive" : "Negative");
    }

    save_nn(nn, "nn.bin");
    
    // Free memory
    free_nn(nn);
    
    return 0;
}