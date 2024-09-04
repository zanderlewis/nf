#include "nf.h"

int main() {
    NeuralNetwork *nn = load_nn("nn.bin");
    
    // Example predictions
    const char *test_samples[] = {
        "good movie", "bad acting", "excellent story", "terrible ending",
        "amazing performance", "disappointing film", "exciting and interesting",
        "boring and awful", "wonderful experience", "poor quality"
    };
    int num_test_samples = 10;

    for (int i = 0; i < num_test_samples; i++) {
        float prediction = predict(nn, test_samples[i]);
        printf("Sample: \"%s\", Prediction: %.2f (%s)\n", test_samples[i], prediction, prediction > 0.5 ? "Positive" : "Negative");
    }

    free_nn(nn);
    return 0;
}
